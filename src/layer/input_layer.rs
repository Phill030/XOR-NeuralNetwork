use super::Layer;
use std::{any::Any, fmt::Debug};

#[derive(Debug, Default, Clone)]
pub struct InputLayer {
    pub biases: Vec<f32>,
    pub neuron_values: Vec<f32>,
    pub errors: Vec<f32>,
    pub weights: Vec<f32>,
    pub following_layer: Option<*const Box<dyn Layer>>,
}

impl Layer for InputLayer {
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

    fn set_previous_layer(&mut self, _layer: Option<*const Box<dyn Layer>>) {
        panic!("InputLayer must not have a previous layer")
    }

    fn set_following_layer(&mut self, layer: Option<*const Box<dyn Layer>>) {
        self.following_layer = layer;
    }

    fn train(&mut self, _learning_rate: &f32, _desired: &Vec<f32>) {
        unreachable!("InputLayer cannot be trained")
    }

    fn feed_forward(&mut self) {
        unreachable!("InputLayer cannot be fed forward")
    }

    //
    fn set_input_data(&mut self, input: &Vec<f32>) {
        for i in 0..input.len() {
            self.neuron_values[i] = input[i];
        }
    }
}

impl InputLayer {
    pub fn new(size: usize) -> Self {
        Self {
            // NOT NEEDED
            biases: vec![],
            neuron_values: vec![0.0; size],
            errors: vec![0.0; size],
            // NOT NEEDED
            weights: vec![],
            following_layer: None,
        }
    }
}
