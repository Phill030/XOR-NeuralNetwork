use std::{any::Any, fmt::Debug};

pub mod dense_layer;
pub mod input_layer;
pub mod layer_holder;
pub mod output_layer;

pub trait Layer
where
    Self: Debug,
{
    fn as_any(&self) -> &dyn Any;
    fn biases(&self) -> &Vec<f32>;
    fn neuron_values(&self) -> &Vec<f32>;
    fn errors(&self) -> &Vec<f32>;
    fn weights(&self) -> &Vec<f32>;
    fn clone_box(&self) -> Box<dyn Layer>;

    fn set_previous_layer(&mut self, layer: Option<*const Box<dyn Layer>>);
    fn set_following_layer(&mut self, layer: Option<*const Box<dyn Layer>>);

    fn feed_forward(&mut self) {}
    fn train(&mut self, _learning_rate: &f32, _desired: &Vec<f32>) {}

    //
    fn set_input_data(&mut self, _input: &Vec<f32>) {
        unreachable!()
    }

    fn set_weights(&mut self, _weights: Vec<f32>) {
        unreachable!()
    }
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Box<dyn Layer> {
        self.clone_box()
    }
}
