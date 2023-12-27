use ndarray::{Array, Array2,Array1 , Dim};
use ndarray_rand::rand_distr::{Uniform, WeightedAliasIndex, Weibull};
use ndarray_rand::RandomExt;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use mnist::*;
use std::collections::VecDeque;

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

        // A weight matrix is defined as a 2d matrix with rows = size of the next layer
        // and col = size of the prev layer. So a network with layers 2 3 2
        // would have a first layer matrix of size 3x2
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

    fn feedforward(&self, mut input: Array<f32, Dim<[usize; 2]>>) -> Array<f32, Dim<[usize; 2]>> {
        for (b, w) in self.bias.iter().zip(self.weights.iter()) {
            input = Self::sigmoid(&(w.dot(&input) + b));
        }
        return input;
    }

    fn sigmoid(z: &Array<f32, Dim<[usize; 2]>>) -> Array<f32, Dim<[usize; 2]>> {
        return z.map(|x| 1.0 / (1.0 + x.exp()));
    }

    fn sigmoid_prime(z: &Arr) -> Arr {
        return z.map(|x| (1.0 / (1.0 + x.exp())) * (1.0 - (1.0 / (1.0 + x.exp()))));
    }

    fn SGD(&mut self, train_set: &mut Vec<(Arr, Arr)>, eta: f32, batch_size: usize, epochs: u32) {
        for i in 0..epochs {
            // seems a waste to shuffle? Is this preformant? 
            // consider creating batch from a list of
            // Shuffled incedies
            train_set.shuffle(&mut thread_rng());
            let batches = train_set.chunks(batch_size);

            for batch in batches {
                self.update(batch, eta);
            }
        }
    }

    fn update(&mut self, batch: &[(Arr, Arr)], eta: f32) {
        // for each example in batch
        //    backprop error
        //    sum nablas
        //    divide by batch size
        // for each layer
        //    adjust weights and bias
        let batch_size = batch.len();
        let mut sum_nabla_w: Vec<Arr> = Vec::new();  
        let mut sum_nabla_b: Vec<Arr> = Vec::new();  

        for example in batch {
            let (nabla_w, nabla_b) = self.backprop(example.clone()); // fix this
            
            for i in 0..self.num_layers - 1 {
                // collect nabla_w
                if sum_nabla_w.len() <= i as usize {
                    sum_nabla_w.push(nabla_w[i as usize].clone());
                } else {
                    sum_nabla_w[i as usize] = sum_nabla_w[i as usize].clone() + nabla_w[i as usize].clone();
                }
                // collect nabla_b
                if sum_nabla_b.len() <= i as usize {
                    sum_nabla_b.push(nabla_b[i as usize].clone());
                } else {
                    sum_nabla_b[i as usize] = sum_nabla_b[i as usize].clone() + nabla_b[i as usize].clone();
                }
            }
        }
        // update weights and bias
        for i in 0_usize..(self.num_layers - 1) as usize {
            self.weights[i] = self.weights[i].clone() - (sum_nabla_w[i].clone() * (1/batch_size) as f32) * eta;
            self.bias[i] = self.bias[i].clone() - (sum_nabla_b[i].clone() * (1/batch_size) as f32) * eta;
        }
    }

    fn backprop(&self, (input, y): (Arr, Arr)) -> (Vec<Arr>, Vec<Arr>) {
        // feed forward
        let mut weighted_activations: Vec<Arr> = Vec::new();
        let mut activations: Vec<Arr> = Vec::new();
        let mut activation = input.clone(); // This is def not preformant
        activations.push(activation.clone());
        for (b, w) in self.bias.iter().zip(self.weights.iter()) {
            let weighted_activation = w.dot(&activation) + b;
            activation = Self::sigmoid(&weighted_activation);
            activations.push(activation.clone());
            weighted_activations.push(weighted_activation.clone());
        }
        // calculate error in output (last layer)
        let mut error = (activation  - y) * Self::sigmoid_prime(&weighted_activations.last().expect("weighted activations empty"));

        // propagate error backwards
        let mut nabla_w: VecDeque<Arr> = VecDeque::new();
        let mut nabla_b: VecDeque<Arr> = VecDeque::new();
        nabla_w.push_front(error.dot(&activations[activations.len() - 2].t()));
        nabla_b.push_front(error.clone());
        
        for i in 2..self.num_layers {
            let wpl = self.weights[self.weights.len() - (i - 1) as usize].clone(); 
            let z = weighted_activations[weighted_activations.len() - (i as usize)].clone();
            error = wpl.t().dot(&error) * Self::sigmoid_prime(&z);
            nabla_w.push_front(error.dot(&activations[activations.len() - (i + 1) as usize].t()));
            nabla_b.push_front(error.clone())
        }
        return (nabla_w.into_iter().collect() , nabla_b.into_iter().collect());
    }
}

fn main() {
    let image_size = 28*28;
    let mut net = Network::new(vec![image_size, 25, 25, 10]);
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    let training_image: Vec<Array1<u8>> = trn_img
        .chunks(image_size as usize)
        .map(|chunks| Array1::from(chunks.to_vec()))
        .collect();
    println!("data set length: {}", training_image.len());
    for image in training_image[0].clone() {
         println!("{}", image);
    }    
}
