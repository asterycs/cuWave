#version 330 core

struct Material
{
  vec3 colorDiffuse;
  vec3 colorAmbient;
};

uniform Material material;
uniform sampler2D shadowMap;

in vec3 vnormal;
in vec4 worldPos;

out vec3 color;

void main(){
    vec3 lightDir = vec3(0, 1, 0);
    vec3 lightIntensity = vec3(200, 200, 200);
    float cosTheta = clamp(dot(vnormal, lightDir), 0.f, 1.f);

    vec3 ambient = material.colorAmbient * 0.25f;
    vec3 diffuse = material.colorDiffuse * cosTheta * lightIntensity;

    color = ambient + diffuse;
}
