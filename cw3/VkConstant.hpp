#pragma once

#include <memory>
#include <volk/volk.h>

namespace {
    template<typename T>
    class VkConstant {
    public:
        VkConstant(unsigned int offset, VkShaderStageFlags usage) {
            m_range.size = sizeof(T);
            m_range.offset = offset;
            m_range.stageFlags = usage;
        }

        void bind(
            VkCommandBuffer cmdBuff, 
            VkPipelineLayout pipeLayout,
            T *data){
            vkCmdPushConstants(cmdBuff, pipeLayout, m_range.stageFlags, m_range.offset, m_range.size, data);
        }
    
        VkPushConstantRange m_range{};
    };
};