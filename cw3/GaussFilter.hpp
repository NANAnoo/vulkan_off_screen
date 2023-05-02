#pragma once

namespace {
    template <unsigned int w>
    struct GaussCompressedFilter1D {
    public:
        GaussCompressedFilter1D(float sigma)
        {
            auto gaussFunc = [sigma](float x) {
                const float mPI = 3.14159265358979323846f;
                return (1.f / (sqrt(2.f * mPI) * sigma)) * exp(- (x * x) / (2.f * sigma * sigma));
            };

            // Calculate the weights at size of w
            unsigned int width = getFilterSize();
            const bool isEven = (w % 2 == 0);
            float weights[w];
            float offsets[w];
            
            for (unsigned int i = 0; i < w; i ++) {
                // odd sequence : x [0, 1, 2 ...], gauss(x) = [, , ,]
                // even sequence: x [0.5, 1.5, 2.5 ...], gauss(x) = [, , ,]
                float x = float(i) + int(isEven) * 0.5f;
                weights[i] = gaussFunc(x);
                offsets[i] = x;
            }

            // compress
            if (isEven) {
                // weights
                for (unsigned int i = 0; i < width; i ++) {
                    data[i] = weights[i * 2] + weights[i * 2 + 1];
                }
                // offsets
                for (unsigned int i = 0; i < width; i ++) {
                    unsigned int l = i * 2, r = l + 1;
                    data[i + width] = (weights[l] * offsets[l] + weights[r] * offsets[r]) / data[i];
                }
            } else {
                data[0] = weights[0] / 2.f; // seperate the first one for simplify the calculation in shader
                data[width] = offsets[0];
                for (unsigned int i = 1; i < width; i ++) {
                    data[i] = weights[i * 2 - 1] + weights[i * 2];
                }
                for (unsigned int i = 1; i < width; i ++) {
                    unsigned int l = i * 2 - 1, r = l + 1;
                    data[i + width] = (weights[l] * offsets[l] + weights[r] * offsets[r]) / data[i];
                }
            }
        }
        ~GaussCompressedFilter1D() = default;

        constexpr static unsigned int getFilterSize() {
            return ( (w + 1u) / 2u );
        }

        constexpr static unsigned int getArraySize() {
            return getFilterSize() * 2u;
        }

    private:
        // ADD PADDING
        float data[getArraySize() + (4 - getArraySize() % 4) % 4];
    };

    template<unsigned int w>
    struct GaussFilter1D {
    public:
        GaussFilter1D(float sigma)
        {
            auto gaussFunc = [sigma](float x) {
                const float mPI = 3.14159265358979323846f;
                return (1.f / (sqrt(2.f * mPI) * sigma)) * exp(- (x * x) / (2.f * sigma * sigma));
            };

            // Calculate the weights at size of w
            const bool isEven = (w % 2 == 0);
            
            for (unsigned int i = 0; i < w; i ++) {
                // odd sequence : x [0, 1, 2 ...], gauss(x) = [, , ,]
                // even sequence: x [0.5, 1.5, 2.5 ...], gauss(x) = [, , ,]
                float x = float(i) + int(isEven) * 0.5f;
                data[i] = gaussFunc(x);
                data[i + w] = x;
            }
            if (!isEven) {
                data[0] = data[0] / 2;
            }
        }
        constexpr static unsigned int getArraySize() {
            return w * 2;
        }
        ~GaussFilter1D() = default;

    private:
        // ADD PADDING
        float data[getArraySize() + (4 - getArraySize() % 4) % 4];
    };
}