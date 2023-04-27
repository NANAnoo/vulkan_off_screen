#pragma once
#include <memory>
#include <vector>
#include <volk/volk.h>

#include "../labutils/vkbuffer.hpp"
#include "../labutils/vulkan_context.hpp"
#include "../labutils/allocator.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vulkan_window.hpp"
#include "../labutils/error.hpp"
#include "../labutils/to_string.hpp"

#include "RenderProgram.hpp"

namespace {
    namespace lut = labutils;
    class VkColorAttachments {
    public:
        VkColorAttachments() = default;
        VkColorAttachments(const VkColorAttachments&) = delete;
        VkColorAttachments& operator=(const VkColorAttachments&) = delete;

        VkColorAttachments(VkColorAttachments && other)
        {
            *this = std::move(other);
        }

        VkColorAttachments& operator=(VkColorAttachments && other) {
            if (this == &other) return *this;
            attachments_set = other.attachments_set;
            frameSize = other.frameSize;
            attachment_formats = std::move(other.attachment_formats);
            attachments = std::move(other.attachments);
            attachment_views = std::move(other.attachment_views);
            attachments_layout = std::move(other.attachments_layout);
            other.attachments_set = VK_NULL_HANDLE;
            return *this;
        }

        void bindAttachments(std::vector<VkImageView> &all_attchments) const {
            for (auto &view : attachment_views) {
                all_attchments.emplace_back(view.handle);
            }
        }

        void setFrameSize(VkExtent2D size) {
            frameSize = size;
        }

        void addAttachment(VkFormat format) {
            attachment_formats.push_back(format);
            // set to null
            attachments_layout = lut::DescriptorSetLayout();
        }

        PipeLineGenerator bindPipeline(PipeLineGenerator pipe) {
            return pipe.addDescLayout(attachments_layout.handle);
        }

        void createAttachmentLayout( lut::VulkanContext const& aContext ) 
        {
            std::vector<VkDescriptorSetLayoutBinding> bindings(attachment_formats.size());
            for (int i = 0; i < int(bindings.size()); i ++) {
                bindings[i].binding = i; // bind texture sampler 0
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[i].descriptorCount = 1; 
                bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; 
            }
            
            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = uint32_t(bindings.size());
            layoutInfo.pBindings = bindings.data();

            VkDescriptorSetLayout layout = VK_NULL_HANDLE;
            if( auto const res = vkCreateDescriptorSetLayout( 
                    aContext.device, 
                    &layoutInfo,
                    nullptr, 
                    &layout ); 
                VK_SUCCESS != res )
            {
                throw lut::Error( "Unable to create descriptor set layout\n" 
                    "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
            }
            attachments_layout = lut::DescriptorSetLayout( aContext.device, layout ); 
        }

        void create(
            lut::VulkanWindow const& aWindow,
            lut::Allocator const& aAllocator,
            lut::DescriptorPool const& dPool,
            lut::RenderPass const& aRenderPass,
            VkSampler aSampler) 
        {
            if (attachments_layout.handle == VK_NULL_HANDLE) {
                createAttachmentLayout(aWindow);
            }
            create_attachments(aWindow, aAllocator);
            // write descriptor set
            std::vector<VkDescriptorImageInfo> attachment_infos(attachments.size());
            for (int idx = 0; idx < int(attachment_infos.size()); idx ++) {
                attachment_infos[idx].sampler = aSampler;
                attachment_infos[idx].imageView =  attachment_views[idx].handle;
                attachment_infos[idx].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            attachments_set = lut::alloc_desc_set(aWindow, dPool.handle, attachments_layout.handle);
            VkWriteDescriptorSet desc{};
            desc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            desc.dstSet = attachments_set; 
            desc.dstBinding = 0; //
            desc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            desc.descriptorCount = uint32_t(attachment_infos.size()); 
            desc.pImageInfo = attachment_infos.data(); 
            vkUpdateDescriptorSets( aWindow.device, 1, &desc, 0, nullptr );
        }

        VkDescriptorSet attachments_set = VK_NULL_HANDLE;

        VkImageView operator[](int idx) const {
            return this->attachment_views[idx].handle;
        }

    private:
        VkExtent2D frameSize;
        std::vector<VkFormat> attachment_formats;
        std::vector<lut::Image> attachments;
        std::vector<lut::ImageView> attachment_views;

        lut::DescriptorSetLayout attachments_layout;
        
        // create attchment images
        void create_attachments( 
            lut::VulkanWindow const& aWindow,
        	lut::Allocator const& aAllocator) 
        { 
            attachments.clear();
            attachment_views.clear();
            for (auto &format : attachment_formats) {
                VkImageCreateInfo imageInfo{};
                imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                imageInfo.imageType = VK_IMAGE_TYPE_2D; 
                imageInfo.format = format;
                imageInfo.extent.width = frameSize.width;
                imageInfo.extent.height = frameSize.height;
                imageInfo.extent.depth = 1; 
                imageInfo.mipLevels = 1;
                imageInfo.arrayLayers = 1;
                imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; 
                imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
                imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 

                VmaAllocationCreateInfo allocInfo{};
                allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; 
                VkImage image = VK_NULL_HANDLE;
                VmaAllocation allocation = VK_NULL_HANDLE; 
                if( auto const res = vmaCreateImage( aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr ); 
                    VK_SUCCESS != res )
                {
                    throw lut::Error( "Unable to allocate screen buffer image.\n"
                        "vmaCreateImage() returned %s", lut::to_string(res).c_str());
                }

                lut::Image screenImage( aAllocator.allocator, image, allocation );

                // Create the image view
                VkImageViewCreateInfo viewInfo{}; 
                viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewInfo.image = screenImage.image; 
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = format;
                viewInfo.components = VkComponentMapping{
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY
                };
                viewInfo.subresourceRange = VkImageSubresourceRange{
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1};
                
                VkImageView view = VK_NULL_HANDLE;

                if( auto const res = vkCreateImageView( aWindow.device, &viewInfo, nullptr, &view );
                    VK_SUCCESS != res )
                {
                    throw lut::Error( "Unable to create image view\n" 
                        "vkCreateImageView() returned %s", lut::to_string(res).c_str() );
                }
                attachments.push_back(std::move(screenImage));
                attachment_views.push_back(lut::ImageView( aWindow.device, view ));
            }
        }
    };
}