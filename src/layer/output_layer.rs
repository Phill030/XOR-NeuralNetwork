use super::Layer;
use crate::math::{activation, activation_deriv, ActivationType};
use rand::{rngs::ThreadRng, Rng};
use std::{any::Any, fmt::Debug};

#[derive(Debug, Default, Clone)]
pub struct OutputLayer {
    pub biases: Vec<f32>,
    pub neuron_values: Vec<f32>,
    pub errors: Vec<f32>,
    pub weights: Vec<f32>,
    pub previous_layer: Option<*const Box<dyn Layer>>,
}

impl Layer for OutputLayer {
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
        if let Some(previous_hidden) = layer {
            self.previous_layer = Some(previous_hidden)
        } else {
            panic!("Previous layer must not be None")
        }
    }

    fn set_following_layer(&mut self, _layer: Option<*const Box<dyn Layer>>) {
        panic!("OutputLayer must not have a following layer")
    }

    fn feed_forward(&mut self) {
        if let Some(previous) = self.previous_layer {
            let previous_layer = unsafe { &*previous };

            let previous_layer_size = previous_layer.neuron_values().len();

            for idx in 0..self.neuron_values.len() {
                let mut sum = 0.0;
                let weight_index = idx * previous_layer_size;

                for j in 0..previous_layer_size {
                    sum += previous_layer.neuron_values()[j] * self.weights[weight_index + j];
                }

                self.neuron_values[idx] = activation(sum + self.biases[idx], ActivationType::Sigmoid);
            }
        } else {
            panic!("Previous layer must not be None")
        }
    }

    fn train(&mut self, learning_rate: &f32, desired: &Vec<f32>) {
        for idx in 0..self.neuron_values.len() {
            self.errors[idx] = desired[idx] - self.neuron_values[idx];
        }

        if let Some(previous) = self.previous_layer {
            let previous_layer = unsafe { &*previous };

            let previous_layer_size = previous_layer.neuron_values().len();

            for idx in 0..self.neuron_values.len() {
                let deriv_neuron_val =
                    learning_rate * self.errors[idx] * activation_deriv(self.neuron_values[idx], ActivationType::Sigmoid);
                let weight_index = idx * previous_layer_size;

                for j in 0..previous_layer_size {
                    self.weights[weight_index + j] += deriv_neuron_val * previous_layer.neuron_values()[j];
                }

                self.biases[idx] += learning_rate * self.errors[idx] * activation_deriv(self.neuron_values[idx], ActivationType::Sigmoid);
            }
        } else {
            panic!("Previous layer must not be None")
        }
    }

    fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights;
    }
}

impl OutputLayer {
    pub fn new(size: usize, rng: &mut ThreadRng) -> Self {
        Self {
            biases: (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            neuron_values: vec![0.0; size],
            errors: vec![0.0; size],
            weights: vec![],
            previous_layer: None,
        }
    }
}
