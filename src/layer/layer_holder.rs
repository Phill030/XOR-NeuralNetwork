use super::Layer;
use crate::layer::{hidden_layer::HiddenLayer, input_layer::InputLayer, output_layer::OutputLayer};
use rand::{rngs::ThreadRng, Rng};

#[derive(Debug)]
pub struct LayerHolder {
    pub layers: Vec<Box<dyn Layer>>,
    rng: ThreadRng,
}

impl LayerHolder {
    pub fn new(rng: &ThreadRng) -> Self {
        Self {
            layers: Vec::new(),
            rng: rng.clone(),
        }
    }

    pub fn add_layer<T: Layer + 'static>(mut self, layer: T) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn build(mut self) -> Self {
        assert!(self.layers.len() > 2, "There must be at least 3 layers!");
        assert!(self.layers[0].as_any().is::<InputLayer>(), "The first layer must be an InputLayer!");
        assert!(
            self.layers[self.layers.len() - 1].as_any().is::<OutputLayer>(),
            "The last layer must be an OutputLayer!"
        );

        for i in 1..self.layers.len() {
            let (prev_layer, current_layer) = self.layers.split_at_mut(i);
            let prev_layer = &mut prev_layer[i - 1];
            let current_layer = &mut current_layer[0];

            if current_layer.as_any().is::<HiddenLayer>() || current_layer.as_any().is::<OutputLayer>() {
                current_layer.set_weights(
                    (0..current_layer.neuron_values().len() * prev_layer.neuron_values().len())
                        .map(|_| self.rng.gen_range(-1.0..1.0))
                        .collect(),
                );
            }

            prev_layer.set_following_layer(Some(current_layer));
            current_layer.set_previous_layer(Some(prev_layer));
        }

        self
    }

    pub fn train(&mut self, input: &Vec<f32>, desired: &Vec<f32>, learning_rate: &f32) -> &mut Self {
        self.feed_forward(&input);

        for i in (1..self.layers.len()).rev() {
            self.layers[i].train(learning_rate, &desired);
        }

        self
    }

    pub fn feed_forward(&mut self, input: &Vec<f32>) -> &Vec<f32> {
        if let Some(input_layer) = self.layers.first_mut() {
            input_layer.set_input_data(input);
        }

        for i in 1..self.layers.len() {
            self.layers[i].feed_forward();
        }

        self.layers[self.layers.len() - 1].neuron_values()
    }
}
