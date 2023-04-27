#pragma once

#include <cstdint>
#include <volk/volk.h>
#include <vk_mem_alloc.h>

#include <utility>

#include <cassert>
#include <unordered_map>
#include <vector>
#include <string>

#include "allocator.hpp"
#include "vulkan/vulkan_core.h"
#include "vulkan_context.hpp"
#include "glm/vec4.hpp"
#include "vkbuffer.hpp"

namespace labutils
{
	class Image
	{
		public:
			Image() noexcept, ~Image();

			explicit Image( VmaAllocator, VkImage = VK_NULL_HANDLE, VmaAllocation = VK_NULL_HANDLE) noexcept;

			Image( Image const& ) = delete;
			Image& operator= (Image const&) = delete;

			Image( Image&& ) noexcept;
			Image& operator = (Image&&) noexcept;

		public:
			VkImage image = VK_NULL_HANDLE;
			VmaAllocation allocation = VK_NULL_HANDLE;

		private:
			VmaAllocator mAllocator = VK_NULL_HANDLE;
	};

	// load multiple textures in a single command buffer
	std::unordered_map<std::string, Image> loadTextures(VulkanContext const&,
							VkCommandPool const&,
                            Allocator const&, 
                            std::vector<std::string> const&);
	
	std::tuple<Image, Buffer>  load_dummy_texture( VulkanContext const&, VkCommandBuffer, Allocator const&, glm::vec4 const& aColor = glm::vec4( 1.0f ));

	Image load_image_texture2d( char const* aPattern, VulkanContext const&, VkCommandPool, Allocator const& );

	Image create_image_texture2d( Allocator const&, std::uint32_t aWidth, std::uint32_t aHeight, VkFormat, VkImageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT );

	std::uint32_t compute_mip_level_count( std::uint32_t aWidth, std::uint32_t aHeight );
}
