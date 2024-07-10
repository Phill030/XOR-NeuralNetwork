use std::f32::EPSILON;

use layer::{dense_layer::DenseLayer, input_layer::InputLayer, layer_holder::LayerHolder, output_layer::OutputLayer};
use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    drawing::IntoDrawingArea,
    element::PathElement,
    series::LineSeries,
    style::{Color, IntoFont, RED, WHITE},
};

pub mod layer;
pub mod math;

fn main() {
    let mut rng = rand::thread_rng();
    let mut layer_holder = LayerHolder::new(&rng)
        .add_layer(InputLayer::new(2))
        .add_layer(DenseLayer::new(4, &mut rng))
        .add_layer(OutputLayer::new(1, &mut rng))
        .build();

    let inputs: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let desired: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let losses = train(&mut layer_holder, &inputs, &desired, 0.01, 20000);

    let predictions = layer_holder.feed_forward(&inputs[0]);
    println!("Predictions: {predictions:#?}");

    draw_loss_plot(&losses).expect("Drawing error");
}

fn train(layer_holder: &mut LayerHolder, inputs: &Vec<Vec<f32>>, desired: &Vec<Vec<f32>>, learning_rate: f32, epochs: usize) -> Vec<f32> {
    let mut losses = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for j in 0..inputs.len() {
            let input = &inputs[j];
            let desired = &desired[j];

            let predictions = layer_holder.feed_forward(input);
            let loss = calculate_cross_entropy_loss(&predictions, desired);
            total_loss += loss;

            layer_holder.train(input, desired, &learning_rate);
        }

        total_loss /= inputs.len() as f32;
        losses.push(total_loss);

        println!("Epoch: {}/{epochs}, Loss: {total_loss}", epoch + 1);
    }

    losses
}

fn calculate_cross_entropy_loss(predictions: &Vec<f32>, target: &Vec<f32>) -> f32 {
    predictions
        .iter()
        .zip(target.iter())
        .map(|(p, t)| -t * (p + EPSILON).ln() - (1.0 - t) * (1.0 - p + EPSILON).ln())
        .sum()
}

fn draw_loss_plot(losses: &Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("loss_plot.png", (800, 800)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Training Loss Over Epochs", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0..losses.len(),
            losses.iter().cloned().fold(0. / 0., f32::min)..losses.iter().cloned().fold(0. / 0., f32::max),
        )?;
    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(x, &y)| (x, y)),
            RED.stroke_width(2),
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED.stroke_width(3)));

    chart.configure_series_labels().background_style(&WHITE).draw()?;

    Ok(())
}
