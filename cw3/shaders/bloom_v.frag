#version 460

layout( location = 0 ) in vec2 v2f_tc;

layout( location = 0 ) out vec4 oColor;

layout( set = 0, binding = 0 ) uniform sampler2D screenTexture;

layout(std140, set = 1, binding = 0 ) buffer readonly Filter {
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
    // calculate intermedium bloom texture
    vec2 pixel_size = 1.0 / textureSize(screenTexture, 0);
    vec2 center_tc = v2f_tc + pixel_size * 0.5;
    // convolution
    vec3 color = vec3(0);
    for (int i = 0; i < uConstants.filter_width; i ++) {
        vec2 offset = vec2(1.f, getData(i + uConstants.filter_width)) * pixel_size;
        float weight = getData(i);
        vec2 tc = center_tc + offset;
        vec3 sample_color = texture(screenTexture, tc).rgb;
        if (sample_color.r > 1.0 || sample_color.g > 1.0 || sample_color.b > 1.0) {
            color = color + sample_color * weight;
        }
        tc = center_tc - offset;
        sample_color = texture(screenTexture, tc).rgb;
        if (sample_color.r > 1.0 || sample_color.g > 1.0 || sample_color.b > 1.0) {
            color = color + sample_color * weight;
        }
    }

    oColor = vec4(color, 1.f);
}