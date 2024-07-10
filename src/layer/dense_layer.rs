use super::Layer;
use crate::math::{activation, activation_deriv, ActivationType};
use rand::{rngs::ThreadRng, Rng};
use std::{any::Any, fmt::Debug};

#[derive(Debug, Default, Clone)]
pub struct DenseLayer {
    pub biases: Vec<f32>,
    pub neuron_values: Vec<f32>,
    pub errors: Vec<f32>,
    pub weights: Vec<f32>,
    pub previous_layer: Option<*const Box<dyn Layer>>,
    pub following_layer: Option<*const Box<dyn Layer>>,
}

impl Layer for DenseLayer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn biases(&self) -> &Vec<f32> {
        &self.biases
    }
    fn neuron_values(&self) -> &Vec<f32> {
        &self.neuron_values
    }

    fn errors(&self) -> &Vec<f32> {
        &self.errors
    }
    fn weights(&self) -> &Vec<f32> {
        &self.weights
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn set_previous_layer(&mut self, layer: Option<*const Box<dyn Layer>>) {
        self.previous_layer = layer;
    }

    fn set_following_layer(&mut self, layer: Option<*const Box<dyn Layer>>) {
        self.following_layer = layer;
    }

    fn feed_forward(&mut self) {
        if let Some(previous) = self.previous_layer {
            let previous_layer = unsafe { &*previous };

            let previous_layer_size = previous_layer.neuron_values().len();

            for i in 0..self.neuron_values.len() {
                let mut sum = 0.0;
                let weight_index = i * previous_layer_size;

                for j in 0..previous_layer_size {
                    sum += previous_layer.neuron_values()[j] * self.weights[weight_index + j];
                }

                self.neuron_values[i] = activation(sum + self.biases[i], ActivationType::Sigmoid);
            }
        } else {
            panic!("Previous layer must not be None")
        }
    }

    fn train(&mut self, learning_rate: &f32, _desired: &Vec<f32>) {
        if let Some(previous) = self.previous_layer {
            let previous_layer = unsafe { &*previous };

            if let Some(following) = self.following_layer {
                let following_layer = unsafe { &*following };

                let previous_layer_size = previous_layer.neuron_values().len();
                let following_layer_size = following_layer.neuron_values().len();

                for idx in 0..self.neuron_values.len() {
                    let mut err = 0.0;
                    let index = idx * previous_layer_size;

                    for j in 0..following_layer_size {
                        err += following_layer.errors()[j] * following_layer.weights()[j * self.neuron_values.len() + idx];
                    }

                    let mut error = err * activation_deriv(self.neuron_values[idx], ActivationType::Sigmoid);
                    self.errors[idx] = error;

                    error *= learning_rate;

                    for j in 0..previous_layer_size {
                        self.weights[index + j] += error * previous_layer.neuron_values()[j];
                    }

                    self.biases[idx] += error;
                }
            } else {
                panic!("Following layer must not be None")
            }
        } else {
            panic!("Previous layer must not be None")
        }
    }

    fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights;
    }
}

impl DenseLayer {
    pub fn new(size: usize, rng: &mut ThreadRng) -> Self {
        Self {
            biases: (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            neuron_values: vec![0.0; size],
            errors: vec![0.0; size],
            weights: vec![],
            following_layer: None,
            previous_layer: None,
        }
    }
}
