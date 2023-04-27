#pragma once


#include <memory>
#include <volk/volk.h>

#include "../labutils/vkbuffer.hpp"
#include "../labutils/vulkan_context.hpp"
#include "../labutils/allocator.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/error.hpp"
#include "../labutils/to_string.hpp"

namespace {
    namespace lut = labutils;
    template<typename Type>
    class VkBO {
    public:
        VkBO() = delete;
        VkBO(
            lut::VulkanContext const& aContext, 
            lut::Allocator const& allocator,
            lut::DescriptorPool const& dPool, 
            VkShaderStageFlags stages,
            VkDescriptorType dDype,
            VkBufferUsageFlagBits usageBit,
            VkAccessFlagBits bit,
            unsigned int binding = 0) : accessBit(bit)
        {
            VkDescriptorSetLayoutBinding bindings[1]{};
            bindings[0].binding = binding;
            bindings[0].descriptorType = dDype;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = stages;

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
            layoutInfo.pBindings = bindings;

            VkDescriptorSetLayout aLayout = VK_NULL_HANDLE;
            if (auto const res = vkCreateDescriptorSetLayout(aContext.device, &layoutInfo, nullptr, &aLayout);
                VK_SUCCESS != res) {
                throw lut::Error("Unable to create descriptor set layout\n"
                    "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
            }
            layout = lut::DescriptorSetLayout(aContext.device, aLayout);

            // create  buffer
            ubo = lut::create_buffer(
                allocator, 
                sizeof(Type),
                usageBit | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY
            );
            // allocate descriptor set for buffer
            set = lut::alloc_desc_set( aContext, dPool.handle, layout.handle );

            // initialize descriptor set with vkUpdateDescriptorSets
            {
                VkWriteDescriptorSet desc[1]{};

                VkDescriptorBufferInfo unoInfo{};
                unoInfo.buffer = ubo.buffer;
                unoInfo.range = VK_WHOLE_SIZE;

                desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                desc[0].dstSet = set;
                desc[0].dstBinding = binding;
                desc[0].descriptorType = dDype;
                desc[0].descriptorCount = 1;
                desc[0].pBufferInfo = &unoInfo;

                constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
                vkUpdateDescriptorSets(aContext.device, numSets, desc, 0, nullptr);
            }
        }

        // allow move noly
        VkBO(const VkBO &) = delete;
        VkBO& operator=(const VkBO&) = delete;
        VkBO(VkBO &&) = default;
        VkBO& operator=(VkBO &&) = default;

        // upload data
        void upload(VkCommandBuffer aCmdBuff, VkPipelineStageFlags stages) {
            // update data
		    lut::buffer_barrier(aCmdBuff, ubo.buffer, 
				accessBit, 
				VK_ACCESS_TRANSFER_WRITE_BIT,
				stages,
				VK_PIPELINE_STAGE_TRANSFER_BIT
			);
            vkCmdUpdateBuffer(aCmdBuff, ubo.buffer, 0, sizeof(Type), data.get());
            lut::buffer_barrier(aCmdBuff,
                ubo.buffer,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                accessBit,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                stages
            );
        }

        lut::Buffer ubo;
        VkDescriptorSet set;
        lut::DescriptorSetLayout layout;
        std::unique_ptr<Type> data;
    private:
        VkAccessFlagBits accessBit;
    };

    template <typename Type>
    class VkUBO : public VkBO<Type> {
    public:
        VkUBO(lut::VulkanContext const& aContext, 
            lut::Allocator const& allocator,
            lut::DescriptorPool const& dPool, 
            VkShaderStageFlags stages,
            unsigned int binding = 0) : 
                VkBO<Type>(aContext, allocator, dPool, stages, 
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_ACCESS_UNIFORM_READ_BIT,
                        binding) 
            {
            }
    };

    template <typename Type>
    class VkSSBO : public VkBO<Type> {
    public:
        VkSSBO(lut::VulkanContext const& aContext, 
            lut::Allocator const& allocator,
            lut::DescriptorPool const& dPool, 
            VkShaderStageFlags stages,
            unsigned int binding = 0) : 
                VkBO<Type>(aContext, allocator, dPool, stages, 
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    binding) 
            {
            }
    };
}