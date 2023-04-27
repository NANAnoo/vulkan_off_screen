#include "vkimage.hpp"

#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>
#include <utility>
#include <algorithm>

#include <cstdio>
#include <cassert>
#include <cstring> // for std::memcpy()

#include <stb_image.h>

#include "error.hpp"
#include "vkutil.hpp"
#include "vkbuffer.hpp"
#include "to_string.hpp"
#include "vulkan/vulkan_core.h"


namespace
{
	// Unfortunately, std::countl_zero() isn't available in C++17; it was added
	// in C++20. This provides a fallback implementation. Unlike C++20, this
	// returns a std::uint32_t and not a signed int.
	//
	// See https://graphics.stanford.edu/~seander/bithacks.html for this and
	// other methods like it.
	//
	// Note: that this is unlikely to be the most efficient implementation on
	// most processors. Many instruction sets have dedicated instructions for
	// this operation. E.g., lzcnt (x86 ABM/BMI), bsr (x86).
	inline 
	std::uint32_t countl_zero_( std::uint32_t aX )
	{
		if( !aX ) return 32;

		uint32_t res = 0;

		if( !(aX & 0xffff0000) ) (res += 16), (aX <<= 16);
		if( !(aX & 0xff000000) ) (res +=  8), (aX <<=  8);
		if( !(aX & 0xf0000000) ) (res +=  4), (aX <<=  4);
		if( !(aX & 0xc0000000) ) (res +=  2), (aX <<=  2);
		if( !(aX & 0x80000000) ) (res +=  1);

		return res;
	}
}

namespace labutils
{
	Image::Image() noexcept = default;

	Image::~Image()
	{
		if( VK_NULL_HANDLE != image )
		{
			assert( VK_NULL_HANDLE != mAllocator );
			assert( VK_NULL_HANDLE != allocation );
			vmaDestroyImage( mAllocator, image, allocation );
		}
	}

	Image::Image( VmaAllocator aAllocator, VkImage aImage, VmaAllocation aAllocation ) noexcept
		: image( aImage )
		, allocation( aAllocation )
		, mAllocator( aAllocator )
	{}

	Image::Image( Image&& aOther ) noexcept
		: image( std::exchange( aOther.image, VK_NULL_HANDLE ) )
		, allocation( std::exchange( aOther.allocation, VK_NULL_HANDLE ) )
		, mAllocator( std::exchange( aOther.mAllocator, VK_NULL_HANDLE ) )
	{}
	Image& Image::operator=( Image&& aOther ) noexcept
	{
		std::swap( image, aOther.image );
		std::swap( allocation, aOther.allocation );
		std::swap( mAllocator, aOther.mAllocator );
		return *this;
	}
}

namespace labutils
{	
std::tuple<Image, Buffer> loadImageHelper(char const* aPattern, 
							VulkanContext const& aContext, 
							VkCommandBuffer cbuff,
							Allocator const& aAllocator) 
	{
		stbi_set_flip_vertically_on_load(true);
		// Load base image
		int width, height, channels;
		stbi_uc *data = stbi_load(aPattern, &width, &height, &channels, STBI_rgb_alpha);

		if (!data) {
			throw Error( "%s: unable to load texture base image (%s)", aPattern, 0, 
				stbi_failure_reason() );
		}

		auto const u_width= std::uint32_t(width);
		auto const u_height= std::uint32_t(height);

		// Create staging buffer and copy image data to it
		auto const sizeInBytes = u_width * u_height * 4;

		auto staging = create_buffer(aAllocator, sizeInBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		void* sptr = nullptr; 
		if( auto const res = vmaMapMemory( aAllocator.allocator, staging.allocation, &sptr )
			; VK_SUCCESS != res ) {
			throw Error( "Mapping memory for writing\n" 
				"vmaMapMemory() returned %s", to_string(res).c_str());
		}
		std::memcpy( sptr, data, sizeInBytes ); 
		vmaUnmapMemory( aAllocator.allocator, staging.allocation );

		// Free image data
		stbi_image_free( data );

		// Create image
		auto ret = create_image_texture2d( aAllocator, u_width, u_height, VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT );

			// Transition whole image layout
		auto const mipLevels = compute_mip_level_count( u_width, u_height );

		image_barrier(cbuff, ret.image, 0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, 
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT, 
				0, mipLevels,
				0, 1
			});

		// Upload data from staging buffer to image
		VkBufferImageCopy copy; 
		copy.bufferOffset = 0;
		copy.bufferRowLength = 0; 
		copy.bufferImageHeight = 0;
		copy.imageSubresource = VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			0, 1
		};
		copy.imageOffset = VkOffset3D{ 0, 0, 0 };
		copy.imageExtent = VkExtent3D{ u_width, u_height, 1 };
		vkCmdCopyBufferToImage( cbuff, staging.buffer, ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy );

		// Transition base level to TRANSFER SRC OPTIMAL
		image_barrier( cbuff, ret.image, 
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, 1, 0, 1
			}
		);

		// Process all mipmap levels
		for( std::uint32_t level = 1; level < mipLevels; ++level ) {
			// Blit previous mipmap level (=level-1) to the current level.
			VkImageBlit blit{};
			blit.srcSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT, 
				level - 1,
				0, 1};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { std::int32_t(width), std::int32_t(height), 1 };

			// Next mip level
			width >>= 1; if( width == 0 ) width = 1;
			height >>= 1; if( height == 0 ) height = 1;

			blit.dstSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT, 
				level, 
				0, 1
			};
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { std::int32_t(width), std::int32_t(height), 1 };
			vkCmdBlitImage( cbuff, 
				ret.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
				ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
				1, &blit, 
				VK_FILTER_LINEAR
			);

			// Transition mip level to TRANSFER SRC OPTIMAL for the next iteration. 
			image_barrier( cbuff, ret.image, 
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
				VK_PIPELINE_STAGE_TRANSFER_BIT, 
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VkImageSubresourceRange{
					VK_IMAGE_ASPECT_COLOR_BIT,
					level, 1, 0, 1
				}
			);
		}

		// Whole image is currently in the TRANSFER SRC OPTIMAL layout. To use the
		// image as a texture from which we sample, it must be in the
		// SHADER READ ONLY OPTIMAL layout. 
		image_barrier( cbuff, ret.image, 
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, mipLevels, 0, 1
			}
		);
		return {std::move(ret), std::move(staging)};
	}

	Image load_image_texture2d( char const* aPattern, VulkanContext const& aContext, VkCommandPool aCmdPool, Allocator const& aAllocator )
	{
		// Flip images vertically by default.
		stbi_set_flip_vertically_on_load(true);

		// Load base image
		int width, height, channels;
		stbi_uc *data = stbi_load(aPattern, &width, &height, &channels, STBI_rgb_alpha);

		if (!data) {
			throw Error( "%s: unable to load texture base image (%s)", aPattern, 0, 
				stbi_failure_reason() );
		}

		auto const u_width= std::uint32_t(width);
		auto const u_height= std::uint32_t(height);

		// Create staging buffer and copy image data to it
		auto const sizeInBytes = u_width * u_height * 4;

		auto staging = create_buffer(aAllocator, sizeInBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		void* sptr = nullptr; 
		if( auto const res = vmaMapMemory( aAllocator.allocator, staging.allocation, &sptr )
			; VK_SUCCESS != res ) {
			throw Error( "Mapping memory for writing\n" 
				"vmaMapMemory() returned %s", to_string(res).c_str());
		}
		std::memcpy( sptr, data, sizeInBytes ); 
		vmaUnmapMemory( aAllocator.allocator, staging.allocation );

		// Free image data
		stbi_image_free( data );

		// Create image
		Image ret = create_image_texture2d( aAllocator, u_width, u_height, VK_FORMAT_R8G8B8A8_SRGB,
		 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT );

		// Create command buffer for data upload and begin recording
		VkCommandBuffer cbuff = alloc_command_buffer( aContext, aCmdPool );

		VkCommandBufferBeginInfo beginInfo{}; 
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0; 
		beginInfo.pInheritanceInfo = nullptr; 
		if( auto const res = vkBeginCommandBuffer( cbuff, &beginInfo ); VK_SUCCESS != res ) {
			throw Error( "Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", to_string(res).c_str());
		}

		// Transition whole image layout
		auto const mipLevels = compute_mip_level_count( u_width, u_height );

		image_barrier(cbuff, ret.image, 0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, 
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT, 
				0, mipLevels,
				0, 1
			});

		// Upload data from staging buffer to image
		VkBufferImageCopy copy; 
		copy.bufferOffset = 0;
		copy.bufferRowLength = 0; 
		copy.bufferImageHeight = 0;
		copy.imageSubresource = VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			0, 1
		};
		copy.imageOffset = VkOffset3D{ 0, 0, 0 };
		copy.imageExtent = VkExtent3D{ u_width, u_height, 1 };
		vkCmdCopyBufferToImage( cbuff, staging.buffer, ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy );

		// Transition base level to TRANSFER SRC OPTIMAL
		image_barrier( cbuff, ret.image, 
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, 1, 0, 1
			}
		);

		// Process all mipmap levels
		for( std::uint32_t level = 1; level < mipLevels; ++level ) {
			// Blit previous mipmap level (=level-1) to the current level.
			VkImageBlit blit{};
			blit.srcSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT, 
				level - 1,
				0, 1};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { std::int32_t(width), std::int32_t(height), 1 };

			// Next mip level
			width >>= 1; if( width == 0 ) width = 1;
			height >>= 1; if( height == 0 ) height = 1;

			blit.dstSubresource = VkImageSubresourceLayers{
				VK_IMAGE_ASPECT_COLOR_BIT, 
				level, 
				0, 1
			};
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { std::int32_t(width), std::int32_t(height), 1 };
			vkCmdBlitImage( cbuff, 
				ret.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
				ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
				1, &blit, 
				VK_FILTER_LINEAR
			);

			// Transition mip level to TRANSFER SRC OPTIMAL for the next iteration. 
			image_barrier( cbuff, ret.image, 
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
				VK_PIPELINE_STAGE_TRANSFER_BIT, 
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VkImageSubresourceRange{
					VK_IMAGE_ASPECT_COLOR_BIT,
					level, 1, 0, 1
				}
			);
		}

		
		// Whole image is currently in the TRANSFER SRC OPTIMAL layout. To use the
		// image as a texture from which we sample, it must be in the
		// SHADER READ ONLY OPTIMAL layout. 
		image_barrier( cbuff, ret.image, 
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, mipLevels, 0, 1
			}
		);

		// End command recording
		if( auto const res = vkEndCommandBuffer( cbuff ); VK_SUCCESS != res ) 
		{
			throw Error( "Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", to_string(res).c_str()
			);
		}

		// Submit command buffer and wait for commands to complete
		Fence uploadComplete = create_fence( aContext );
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; 
		submitInfo.commandBufferCount = 1; 
		submitInfo.pCommandBuffers = &cbuff;
		if( auto const res = vkQueueSubmit( aContext.graphicsQueue, 1, &submitInfo, 
			uploadComplete.handle ); VK_SUCCESS != res ) 
		{
			throw Error( "Submitting commands\n"
				"vkQueueSubmit() returned %s", to_string(res).c_str());
		}

		if( auto const res = vkWaitForFences(
				aContext.device, 
				1, 
				&uploadComplete.handle,
				VK_TRUE, 
				std::numeric_limits<std::uint64_t>::max() ); 
			VK_SUCCESS != res)
		{
			throw Error( "Waiting for upload to complete\n"
				"vkWaitForFences() returned %s", to_string(res).c_str());
		}
		// return resulting image, free cmd buffer
		vkFreeCommandBuffers( aContext.device, aCmdPool, 1, &cbuff );

		return ret;
	}

	Image create_image_texture2d( Allocator const& aAllocator, std::uint32_t aWidth, std::uint32_t aHeight, VkFormat aFormat, VkImageUsageFlags aUsage )
	{
		auto const mipLevels = compute_mip_level_count(aWidth, aHeight);

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = aFormat;
		imageInfo.extent.width = aWidth;
		imageInfo.extent.height = aHeight;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = aUsage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		// declare the usage of this image
		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		// alloc a buffer to this image: aka allocation
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		if (auto const res = vmaCreateImage(
			aAllocator.allocator, &imageInfo,
			 &allocInfo, &image,
			  &allocation, nullptr);
			VK_SUCCESS != res) 
		{
			throw Error("Umable to alloc image.\n"
				"vmaCreateImage() returned %s", to_string(res).c_str());
		}
		return Image( aAllocator.allocator, image, allocation);
	}


	std::unordered_map<std::string, Image> loadTextures(VulkanContext const& aContext,
							VkCommandPool const& aCmdPool,
                            Allocator const& aAllocator, 
                            std::vector<std::string> const& paths)
	{
		// Create command buffer for data upload and begin recording
		VkCommandBuffer cbuff = alloc_command_buffer( aContext, aCmdPool);
		VkCommandBufferBeginInfo beginInfo{}; 
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0; 
		beginInfo.pInheritanceInfo = nullptr; 
		if( auto const res = vkBeginCommandBuffer( cbuff, &beginInfo ); VK_SUCCESS != res ) {
			throw Error( "Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", to_string(res).c_str());
		}
		std::unordered_map<std::string, std::tuple<Image, Buffer>> tmp;
		for (auto &path : paths) {
			// load & upload all textures in a cmd buffer
			tmp.insert({path, loadImageHelper(path.c_str(), aContext, cbuff, aAllocator)});
		}
		// End command recording
		if( auto const res = vkEndCommandBuffer( cbuff ); VK_SUCCESS != res ) 
		{
			throw Error( "Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", to_string(res).c_str()
			);
		}

		// submit current cmdbuffer
		// Submit command buffer and wait for commands to complete
		Fence uploadComplete = create_fence( aContext );
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; 
		submitInfo.commandBufferCount = 1; 
		submitInfo.pCommandBuffers = &cbuff;
		if( auto const res = vkQueueSubmit( aContext.graphicsQueue, 1, &submitInfo, 
			uploadComplete.handle ); VK_SUCCESS != res ) 
		{
			throw Error( "Submitting commands\n"
				"vkQueueSubmit() returned %s", to_string(res).c_str());
		}

		if( auto const res = vkWaitForFences(
				aContext.device, 
				1, 
				&uploadComplete.handle,
				VK_TRUE, 
				std::numeric_limits<std::uint64_t>::max() ); 
			VK_SUCCESS != res)
		{
			throw Error( "Waiting for upload to complete\n"
				"vkWaitForFences() returned %s", to_string(res).c_str());
		}
		// return resulting image, free cmd buffer
		vkFreeCommandBuffers( aContext.device,aCmdPool, 1, &cbuff );
		std::unordered_map<std::string, Image> res;
		for (auto &[path, tex] : tmp) {
			res[path] = std::move(std::get<0>(tex));
		}
		return res;
	}

	std::tuple<Image, Buffer> load_dummy_texture( VulkanContext const& aContext, VkCommandBuffer cbuff, Allocator const&aAllocator, glm::vec4 const& aColor)
	{
		// Create a 1x1 image with the color
		std::uint8_t const data[4] = {std::uint8_t(aColor.r * 255), std::uint8_t(aColor.g * 255), std::uint8_t(aColor.b * 255), std::uint8_t(aColor.a * 255)};

		// Create a staging buffer
		Buffer staging = create_buffer( aAllocator, 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU );

		// Copy data to staging buffer
		void* mapped = nullptr;
		if( auto const res = vmaMapMemory( aAllocator.allocator, staging.allocation, &mapped ); VK_SUCCESS != res ) {
			throw Error( "Mapping staging buffer memory\n"
				"vmaMapMemory() returned %s", to_string(res).c_str());
		}
		std::memcpy( mapped, data, 4 );
		vmaUnmapMemory( aAllocator.allocator, staging.allocation );

		// Create image
		Image ret = create_image_texture2d( aAllocator, 1, 1, VK_FORMAT_R8G8B8A8_SRGB,
		 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT );

		image_barrier(cbuff, ret.image, 0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, 
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT, 
				0, 1,
				0, 1
			});

		// Upload data from staging buffer to image
		VkBufferImageCopy copy; 
		copy.bufferOffset = 0;
		copy.bufferRowLength = 0; 
		copy.bufferImageHeight = 0;
		copy.imageSubresource = VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			0, 1
		};
		copy.imageOffset = VkOffset3D{ 0, 0, 0 };
		copy.imageExtent = VkExtent3D{ 1, 1, 1 };
		vkCmdCopyBufferToImage( cbuff, staging.buffer, ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy );

		// Transition base level to TRANSFER SRC OPTIMAL
		image_barrier( cbuff, ret.image, 
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, 1, 0, 1
			}
		);

		// Whole image is currently in the TRANSFER SRC OPTIMAL layout. To use the
		// image as a texture from which we sample, it must be in the
		// SHADER READ ONLY OPTIMAL layout. 
		image_barrier( cbuff, ret.image, 
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, 1, 0, 1
			}
		);

		return std::make_tuple(std::move(ret), std::move(staging));
	}

	std::uint32_t compute_mip_level_count( std::uint32_t aWidth, std::uint32_t aHeight )
	{
		std::uint32_t const bits = aWidth | aHeight;
		std::uint32_t const leadingZeros = countl_zero_( bits );
		return 32-leadingZeros;
	}
}
