use std::iter::zip;

pub fn dot_matrices(m1: &Vec<Vec<f32>>, m2: &Vec<Vec<f32>>) -> f32 {
   let sum =  zip(m1, m2).fold(0.0, |sum_m, (v1, v2)| 
              sum_m + zip(v1, v2).fold(0.0, |sum_v,  (c1, c2)| sum_v + (c1 * c2)));
   return sum;
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_matrices() {
        let m1: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![3.0, 9.0, 4.0], vec![8.0, 2.0, 3.0]];
        let m2: Vec<Vec<f32>> = vec![vec![3.0, 9.0, 4.0], vec![3.0, 3.0, 4.0], vec![8.0, 1.0, 5.0]];
        let dot_product = dot_matrices(&m1, &m2);
        assert!(dot_product == 166.0);
    }
}

