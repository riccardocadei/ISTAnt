# [anonymized]Ant

Automatic treatment effect estimation on ecological data with partial labelling.

## Dataset 
Preliminary release at [google drive](https://drive.google.com/drive/folders/1ZTPusp-u3pAtrs2LtA3JUaFXbuqDS7K_?usp=sharing)

### Example

<table align="center">
  <tr>
    <th>No Action</th>
    <th>Grooming (<i>Y2F</i>)</th>
  </tr>
  <tr>
    <td><img src="img/example_nogrooming.gif" alt="GIF 1" width="300" height="300"></td> 
    <td><img src="img/example_grooming.gif" alt="GIF 2" width="300" height="300"></td>
  </tr>
</table>

### Data Distribution

![Outcome distribution](img/outcome_distribution.png)

### Research Question

Identify and estimate:
$$ATE := \mathbb{E}[Y|do(T=2)]- \mathbb{E}[Y|do(T=1)]$$
