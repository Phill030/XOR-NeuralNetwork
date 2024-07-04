pub enum ActivationType {
    Sigmoid,
    Relu,
    Softmax,
    TanH,
    LeakyRelu,
    ELU,
    Swish,
}

pub fn activation(x: f32, activation: ActivationType) -> f32 {
    match activation {
        ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        ActivationType::Relu => x.max(0.0),
        ActivationType::Softmax => (x.exp()) / (1.0 + (x.exp())),
        ActivationType::TanH => x.tanh(),
        ActivationType::LeakyRelu => {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        }
        ActivationType::ELU => {
            if x > 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        ActivationType::Swish => x / (1.0 + (-x).exp()),
    }
}

pub fn activation_deriv(x: f32, activation: ActivationType) -> f32 {
    match activation {
        ActivationType::Sigmoid => x * (1.0 - x),
        ActivationType::Relu => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        ActivationType::Softmax => x * (1.0 - x),
        ActivationType::TanH => 1.0 - x.tanh().powi(2),
        ActivationType::LeakyRelu => {
            if x > 0.0 {
                1.0
            } else {
                0.01
            }
        }
        ActivationType::ELU => {
            if x > 0.0 {
                1.0
            } else {
                x + 1.0
            }
        }
        ActivationType::Swish => {
            let sigma = 1.0 / (1.0 + (-x).exp());
            sigma * (1.0 + x * (1.0 - sigma))
        }
    }
}
