#pragma once
// vulkan vertex buffer object

#include "../labutils/vkutil.hpp"
#include "../labutils/error.hpp"
#include "../labutils/to_string.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/vkimage.hpp"

namespace {
    namespace lut = labutils;

    class VkVBO {
    public:
        VkVBO() = delete;
        VkVBO(
            lut::VulkanContext const& aContext, 
            lut::Allocator const& aAllocator,
            std::size_t dataSize,
            void* data, 
            VkBufferUsageFlagBits usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) : dataSize(dataSize)
        {
            // GPU buffer
            VBO = lut::create_buffer(
                aAllocator,
                dataSize,
                usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY
            );

            // staging buffer
            staging = lut::create_buffer(
                aAllocator,
                dataSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU 
            );

            // copy data
            void* tempBuff = nullptr;
            if (auto const res = vmaMapMemory(
                aAllocator.allocator,
                staging.allocation,
                &tempBuff);
                VK_SUCCESS != res) {
                throw labutils::Error("Mapping memory for writing\n"
                    "vmaMapMemory() returned %s", labutils::to_string(res).c_str());
            }
            memcpy(tempBuff, data, dataSize);
            vmaUnmapMemory(aAllocator.allocator, staging.allocation);
        }

        void upload(VkCommandBuffer uploadCmd) {
            VkBufferCopy copyRegion = {};
            copyRegion.size = dataSize;
            vkCmdCopyBuffer(uploadCmd, staging.buffer, VBO.buffer, 1, &copyRegion);
            lut::buffer_barrier(
            uploadCmd,
            VBO.buffer,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
            );

            // free staging buffer
            staging = lut::Buffer();
        }
        const VkBuffer get() const {
            return VBO.buffer;
        }
    
    protected:
        lut::Buffer VBO;
        lut::Buffer staging;
        std::size_t dataSize;
    };

    class VkIBO : public VkVBO {
    public:
        VkIBO() = delete;
        VkIBO(
            lut::VulkanContext const& aContext,
            lut::Allocator const& aAllocator,
            std::size_t dataSize,
            std::size_t count,
            void* data) : VkVBO(aContext, aAllocator, dataSize, data, VK_BUFFER_USAGE_INDEX_BUFFER_BIT), count(count)
        {
        }

        void bind(VkCommandBuffer cmd) {
            vkCmdBindIndexBuffer(cmd, VBO.buffer, 0, VK_INDEX_TYPE_UINT32);
        }

        void draw(VkCommandBuffer cmd) {
            vkCmdDrawIndexed(cmd, uint32_t(count), 1, 0, 0, 0);
        }
    private:
        std::size_t count;
    };
}