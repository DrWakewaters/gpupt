struct triangle {
    float4 node_0;
    float4 node_1;
    float4 node_2;
    float4 e_1;
    float4 e_2;
    float4 normal;
    float4 t_1;
    float4 t_2;
    float4 color;
    bool is_light;
};

struct ray {
    float4 position;
    float4 direction;
};
