#version 460

layout( location = 0 ) in vec2 v2f_tc;

layout( location = 0 ) out vec4 oColor;

layout( set = 0, binding = 0 ) uniform sampler2D middleTexture;

layout( set = 1, binding = 0 ) uniform sampler2D screenTexture;

layout(std140, set = 2, binding = 0 ) buffer readonly Filter {
    // 2 * length, [weight ..., offset ...]
    vec4 data[];
} uFilter;

layout(push_constant) uniform Constants {
    int filter_width;
} uConstants;

float getData(int index) {
    int v_i = index / 4;
    int offset = index % 4;
    return uFilter.data[v_i][offset];
}

void main()
{
    vec2 pixel_size = 1.0 / textureSize(middleTexture, 0);
    vec2 center_tc = v2f_tc + pixel_size * 0.5;
    // convolution
    // screen color + bloom color * weight[0]
    vec3 color = vec3(0.0);
    for (int i = 0; i < uConstants.filter_width; i ++) {
        vec2 offset = vec2(getData(i + uConstants.filter_width), 1.f) * pixel_size;
        float weight = getData(i);
        vec2 tc = center_tc + offset;
        vec3 sample_color = texture(middleTexture, tc).rgb;
        color = color + sample_color * weight;

        tc = center_tc - offset;
        sample_color = texture(middleTexture, tc).rgb;
        color = color + sample_color * weight;
    }
    color = color + texture(screenTexture, center_tc).rgb;

    oColor = vec4(color / (color + 1.0), 1);
}