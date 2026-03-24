## RL

$$
s_t-state
\\o_t-observation
\\a_t-action
\\c(s_t,a_t)-cost \ function
\\r(s_t,a_t)-reward \ function
$$


$$
r(s,a)\&p(s'|s,a) 
$$
define the Markov Decision Process
$$
M=\{S,T\}
\\
p(s_{t+1}|s_t)——transition \ operator
\\
if  \ \mu_{t,i}=p(s_t=i)\\
then \ \mu_{t+1}=T\mu_t
$$

### Actions

Markov Decision Process
$$
M=\{S,A,T,r\}
\\
\sigma-emission \ probability
$$

### Goal

Chain Rule of Prob.
$$
p_\theta(\mathbf{s}_1,\mathbf{a}_1,\ldots,\mathbf{s}_H,\mathbf{a}_H)=p(\mathbf{s}_1)\prod_{t=1}^H\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)p(\mathbf{s}_{t+1}|\mathbf{s}_t,\mathbf{a}_t)
$$

$$
\theta^\star=\arg\max_\theta\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_tr(\mathbf{s}_t,\mathbf{a}_t)\right]
$$

[Home – Weights & Biases](https://wandb.ai/home)

#### Stationary Distribution Exsit?

### Algorithm

#### Step

- generate samples
- fit the model
- improve the policy

#### Classification

- Policy gradients
- Value-based
- Actor-critic

#### Model-based RL Algo

#### Tradeoffs



#### Off/On Policy

python src/hw1_imitation/train.py --policy_type mse --num_epochs 5 --eval_interval 100