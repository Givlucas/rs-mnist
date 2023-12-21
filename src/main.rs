use ndarray::{Array, Dim};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::prelude::SliceRandom;
use rand::thread_rng;

type Arr = Array<f32, Dim<[usize; 2]>>;

pub struct Network {
    num_layers: u32,
    size: Vec<u32>,
    bias: Vec<Array<f32, Dim<[usize; 2]>>>,
    weights: Vec<Array<f32, Dim<[usize; 2]>>>,
}

impl Network {
    pub fn new(layers: Vec<u32>) -> Self {
        let mut bias = Vec::new();
        let mut weights = Vec::new();

        for y in layers[1..].iter() {
            let layer = Array::random((*y as usize, 1), Uniform::new(0., 1.));
            bias.push(layer);
        }

        for (x, y) in layers[..layers.len()].iter().zip(layers[1..].iter()) {
            let layer = Array::random((*y as usize, *x as usize), Uniform::new(0., 1.));
            weights.push(layer);
        }

        Self {
            num_layers: layers.len() as u32,
            size: layers,
            bias,
            weights,
        }
    }

    fn feedforward(self, mut input: Array<f32, Dim<[usize; 2]>>) -> Array<f32, Dim<[usize; 2]>> {
        for (b, w) in self.bias.iter().zip(self.weights.iter()) {
            input = Self::sigmoid(&(w.dot(&input) + b));
        }
        return input;
    }

    fn sigmoid(z: &Array<f32, Dim<[usize; 2]>>) -> Array<f32, Dim<[usize; 2]>> {
        return z.map(|x| 1.0 / (1.0 + x.exp()));
    }

    fn SGD(self, train_set: &mut Vec<(Arr, Arr)>, eta: u32, batch_size: usize, epochs: u32) {
        for i in 0..epochs {
            train_set.shuffle(&mut thread_rng());
            let batches = train_set.chunks(batch_size);

            for batch in batches {
                Self::update(batch, eta);
            }
        }
    }

    fn update(batch: &[(Arr, Arr)], eta: u32) {}
}

fn main() {
    let mut test = Network::new(vec![2, 3, 5]);
    let input = Array::random((2, 1), Uniform::new(0., 1.));
    let output = test.feedforward(input.clone());
    // let mut test2 = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    // test2.shuffle(&mut thread_rng());
    // println!("{:?}", test2)
    println!("input: {input} \n output: {output}");
}
