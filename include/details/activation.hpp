#ifndef AMSUJNBRSEOVYUBWEASNQBUSARCMLHXXCIMECLOQNQTHJOYOXUQAEVNCJJIFLPCBPXCQKMWOC
#define AMSUJNBRSEOVYUBWEASNQBUSARCMLHXXCIMECLOQNQTHJOYOXUQAEVNCJJIFLPCBPXCQKMWOC

#include "./utility.hpp"

namespace cunn
{
    // all the activation function currently implemented
    //static const std::set<std::string> activation_functions
    static const std::vector<std::string> activation_functions
    {
        "identity",
        "logistic",
        "tanh",
        "arctan",
        "softsign",
        "relu",
        "selu"
        "leaky_relu",
        "softplus",
        "bent_identity",
        "sinusoid",
        "sinc",
        "gaussian"
    };

    //ref: https://en.wikipedia.org/wiki/Activation_function
    static const std::map<std::string,std::string> activation_code
    {
        {
            "identity_activation",
            "[](float x, float& y, float& y){y = x;}"
        },
        {
            "identity_derivative",
            "[](float x, float& y){y = 1.0f;})"
        },

        {
            "logistic_activation",
            "[](float x, float& y){y = 1.0f/(1.0f+expf(-x));}"
        },
        {
            "logistic_derivative",
            "[](float x, float& y){float fx = 1.0f/(1.0f+expf(-x)); y = fx * (1-fx);}"
        },

        {
            "tanh_activation",
            "[](float x, float& y){y = tanhf(x);}"
        },
        {
            "tanh_derivative",
            "[](float x, float& y){float fx = tanhf(x); y = 1.0f - fx*fx;}"
        },

        {
            "arctan_activation",
            "[](float x, float& y){y = atanf(x);}"
        },
        {
            "arctan_derivative",
            "[](float x, float& y){y = 1.0f / (1.0f+x*x);}"
        },

        {
            "softsign_activation",
            "[](float x, float& y){y = x / (1.0f+fabsf(x));}"
        },
        {
            "softsign_derivative",
            "[](floatx){float bs = 1.0f + fabsf(x); y = 1.0f/(bs*bs);}"
        },

        {
            "relu_activation",
            "[](float x, float& y){ if (x>=0.0f) y = x; else y = 0.0f;}"
        },
        {
            "relu_derivative",
            "[](float x, float& y){ if (x>=0.0f) y = 1.0f; else y = 0.0f;}"
        },

        {
            "selu_activation",
            "[](float x, float& y){ if(x>=0.0f) y = 1.67326f*x; else y = 1.758094282f*(expf(x)-1.0f);}"
        },
        {
            "selu_derivative",
            "[](float x, float& y){ if(x>=0.0f) y = 1.67326f; else y = 2.94174883829932f*(expf(x)-1.0f);}"
        },

        {
            "softplus_activation",
            "[](float x, float& y){y = logf(1.0+expf(x));}"
        },
        {
            "softplus_derivative",
            "[](float x, float& y){y = 1.0f/(1.0f+expf(-x));}"
        },

        {
            "leaky_relu_activation",
            "[](float x, float& y){ if (x>=0.0f) y = x; else y = 0.01f*x;}"
        },
        {
            "leaky_relu_derivative",
            "[](float x, float& y){ if(x>=0.0f) y = 1.0f; else y = 0.01f;}"
        },

        {
            "bent_identity_activation",
            "[](float x, float& y){y = x - 0.5f + 0.5f*sqrtf(x*x+1.0f);}"
        },
        {
            "bent_identity_derivative",
            "[](float x, float& y){y = x/(2.0f*sqrtf(x*x+1.0f))+1.0f;}"
        },

        {
            "sinsoid_activation",
            "[](float x, float& y){y = sinf(x);}"
        },
        {
            "sinsoid_derivative",
            "[](float x, float& y){y = cosf(x);}"
        },

        {
            "sinc_activation",
            "[](float x, float& y){ if (fabsf(x)>1.0e-10f) y = sinf(x)/x; else y = 0.0f}"
        },
        {
            "sinc_derivative",
            "[](float x, float& y){ if (fabsf(x)>1.0e-10f) y = ( x * cosf(x) - sinf(x) ) / ( x*x ); else y = 0.0f;}"
        },

        {
            "gaussian_activation",
            "[](float x, float& y){y = expf(-x*x);}"
        },
        {
            "gaussian_derivative",
            "[](float x, float& y){y = -2.0f*x*expf(-x);}"
        }
    };

}

#endif//AMSUJNBRSEOVYUBWEASNQBUSARCMLHXXCIMECLOQNQTHJOYOXUQAEVNCJJIFLPCBPXCQKMWOC
