#version 330 core

struct Material
{
  vec3 colorDiffuse;
  vec3 colorAmbient;
};

uniform Material material;
uniform sampler2D shadowMap;
uniform vec3 cameraWorldPos;

in vec3 vnormal;
in vec4 worldPos;

out vec3 color;

void main(){
    vec3 lightPos = cameraWorldPos;
    vec3 lightIntensity = vec3(1, 1, 1);
    float cosTheta = clamp(dot(normalize(vnormal), -normalize(vec3(worldPos) - lightPos)), 0.f, 1.f);

    vec3 ambient = material.colorAmbient * 0.25f;
    vec3 diffuse = material.colorDiffuse * cosTheta * lightIntensity;

    color = ambient + diffuse;
}
