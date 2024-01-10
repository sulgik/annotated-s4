# <center><h1> The Annotated S4 (한국어)</h1></center>
#
#
# <center>
# <p><a href="https://arxiv.org/abs/2111.00396">Efficiently Modeling Long Sequences with Structured State Spaces</a></p>
# </center>
#
# <center>
# <p> Albert Gu, Karan Goel, and Christopher Ré.</p>
# </center>
# <img src="images/hero.png" width="100%"/>


# *[Sasha Rush](http://rush-nlp.com/) 와 [Sidd Karamcheti](https://www.siddkaramcheti.com/) 의 블로그와 [라이브러리](https://github.com/srush/annotated-s4/) by *, v3

#
# [Structured State Space for Sequence
# Modeling](https://arxiv.org/abs/2111.00396) (S4) 아키텍쳐는
# 시각, 언어 및 오디오에서 매우 긴 시퀀스 모델링 작업에 대한 새로운 접근방식으로, 수만 단계에 걸친 
# 의존성을 담을 수 있는 성능을 보여줍니다. 특히 인상적인 것은
# [Long Range Arena](https://github.com/google-research/long-range-arena) 벤치마크에서의 결과로 
# 최대 **16,000+** 이상의 요소에 대한 시퀀스에서 높은 정확도로 추론할 수 있는 능력을 보여줍니다.

# <img src="images/table.png" width="100%"/>

# 이 논문은 트랜스포머(Transformer)에서 벗어나 중요한 문제 영역에 대해
# 매우 다른 접근 방식을 취하고 있어 상쾌합니다. 그러나,
# 여러 동료들이 모델에 대한 직관을 얻기 어렵다고 사적으로 지적한 바 있습니다.
# 이 블로그 게시물은 직관을 얻기 위한 첫 단계로, 구체적인 코드
# 구현과 S4 논문의 설명을 연결합니다 ([the annotated 
# Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) 스타일).
# 코드와 문해력 있는 설명이
# 모델을 디테일하게 이해하는데 도움이 되기를 바랍니다. 이 블로그를 다 읽으면
# 효율적인 작동 버전의 S4 를 갖게 될 것이며, 이는 훈련 시 CNN 으로 작동할 수 있고,
# 테스트 시에는 효율적인 RNN으로 전환할 수 있습니다. 결과를 미리 보면,
# 표준 GPU 에서 픽셀로부터 이미지를 생성하고 오디오 파형으로부터 직접 소리를 생성할 수 있습니다.
#
# <center> <img src="images/im0.4.png" width="70%">
# <img src='images/speech25.0.png' width='80%'>
# <audio controls>
#  <source src='images/sample25.0.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample25.0.gold.wav' type='audio/wav'>
# </audio>
# </center>

# ## Table of Contents


# * [Part 1: State Space Models] (Modeling)
#     - [Discrete-time SSM: The Recurrent Representation]
#     - [Tangent: A Mechanics Example]
#     - [Training SSMs: The Convolutional Representation]
#     - [An SSM Neural Network.]
# * [Part 1b: HiPPO 로 long-range 의존성 해결]
# * [Part 2: S4 구현] (Advanced)
#     - [Step 1. SSM Generating Functions]
#     - [Step 2: Diagonal Case]
#     - [Step 3: Diagonal Plus Low-Rank]
#     - [Diagonal Plus Low-Rank RNN.]
#     - [Turning HiPPO to DPLR]
#     - [Final Check]
# * [Part 3: S4 in Practice] (NN Implementation)
#     - [S4 CNN / RNN Layer]
#     - [Sampling and Caching]
#     - [Experiments: MNIST]
#     - [Experiments: QuickDraw]
#     - [Experiments: Spoken Digits]
# * [Conclusion]


# <nav id="TOC">

# 이 프로젝트는 [JAX](https://github.com/google/jax/)를 사용하며
# [Flax](https://github.com/google/flax) NN 라이브러리와 함께합니다. 우리는 개인적으로 주로 Torch를 사용하지만,
# JAX 의 함수적 특성은 S4 의 복잡성에 잘 맞습니다. 우리는
# [vmap](https://jax.readthedocs.io/en/latest/jax.html#jax.vmap),
# [scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html),
# 그리고 그들의 [NN
# 친척들](https://flax.readthedocs.io/en/latest/flax.linen.html#module-flax.linen.transforms),
# 그리고 가장 중요하게는
# [jax.jit](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables)
# 을 사용하여 빠르고 효율적인 S4 레이어를 컴파일합니다.

from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve


if __name__ == "__main__":
    # For this tutorial, construct a global JAX rng key
    # But we don't want it when importing as a library
    rng = jax.random.PRNGKey(1)


# ## Part 1: State Space Models

# 시작해봅시다! 우리의 목표는 긴 시퀀스를 효율적으로 모델링하는 것입니다. 
# 이를 위해, 우리는 state space model 을 기반으로 하는 신경망 레이어를 구축할 것입니다. 
# 이 섹션에서는 이 레이어를 사용하여 모델을 구축하고 실행할 수 있게 될 것입니다. 하지만, 기술적인 배경 지식이 필요합니다. 논문의 background 를 통해 함께 알아가 봅시다.


# > [state space model](https://en.wikipedia.org/wiki/State-space_representation) 은
# > 이 간단한 방정식으로 정의됩니다.
# > 이는 1차원 입력 신호 $u(t)$ 를 $N$ 차원 잠재 상태 $x(t)$ 로 매핑한 후,
# > 1차원 출력 신호 $y(t)$ 로 투영합니다.
# $$
#   \begin{aligned}
#     x'(t) &= \boldsymbol{A}x(t) + \boldsymbol{B}u(t) \\
#     y(t) &= \boldsymbol{C}x(t) + \boldsymbol{D}u(t)
#   \end{aligned}
# $$
# > 우리의 목표는
# > SSM 을 심층 시퀀스 모델에서 블랙박스 표현으로 단순하게 사용하는 것이며, 
# > 여기서 $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}, \boldsymbol{D}$ 는
# > 경사하강법으로 학습된 매개변수입니다. 나머지에 대해서는
# > 설명을 위해 매개변수 $\boldsymbol{D}$ 를 생략하겠습니다 (또는 동등하게,
# > $\boldsymbol{D} = 0$ 으로 가정합니다. 왜냐하면 항 $\boldsymbol{D}u$ 는
# > 스킵 커넥션으로 볼 수 있고 계산하기 쉽기 때문입니다).
# >
# > SSM 은 입력 $u(t)$ 를 상태 표현 벡터 $x(t)$ 와 출력 $y(t)$ 로 매핑합니다.
# > 단순화하여, 입력과 출력을 일차원으로 가정하고, 상태 표현은
# > $N$-차원으로 합니다. 첫 번째 방정식은 시간에 따른 $x(t)$의 변화를 정의합니다.

# 우리의 SSM 은 세 개의 행렬 - $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}$ - 로 정의될 것이며,
# 우리는 이것들을 학습할 것입니다. 우선 우리는 임의의 SSM 으로 시작하여 크기를 정의합니다,


def random_SSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C


# ### Discrete-time SSM: The Recurrent Representation
#
# > 이산 입력 시퀀스 $(u_0, u_1, \dots )$ 에 적용하기 위해서
# > 연속함수 $u(t)$ 대신,
# > 입력의 해상도를 나타내는 **스텝 크기** $\Delta$ 를 이용하여
# > SSM 을 이산화해야 합니다. 개념적으로, 입력 $u_k$ ($= u(k \Delta)$) 는
# > 내재적인 연속신호 $u(t)$ 를 샘플링하는 것으로 볼 수 있습니다.
# >
# > continuous-time SSM 을 이산화하기 위해, 우리는
# > [bilinear method](https://en.wikipedia.org/wiki/Bilinear_transform) 를 사용합니다. 이 방법은
# > 상태 행렬 $\boldsymbol{A}$를 근사 $\boldsymbol{\overline{A}}$로 변환합니다. 이산 SSM은:
# $$
# \begin{aligned}
#   \boldsymbol{\overline{A}} &= (\boldsymbol{I} - \Delta/2 \cdot \boldsymbol{A})^{-1}(\boldsymbol{I} + \Delta/2 \cdot \boldsymbol{A}) \\
#   \boldsymbol{\overline{B}} &= (\boldsymbol{I} - \Delta/2 \cdot \boldsymbol{A})^{-1} \Delta \boldsymbol{B} \\
#   \boldsymbol{\overline{C}} &= \boldsymbol{C}\\
# \end{aligned}
# $$


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C

# > 이 방정식은 이제 함수에서 함수로의 매핑이 아닌 *시퀀스-투-시퀀스* 매핑 $u_k \mapsto y_k$ 입니다.
# > 또한, 상태 방정식은 이제 $x_k$ 에서의 반복으로, 이산 SSM 이 RNN 처럼 계산될 수 있게 합니다.
# > 구체적으로, $x_k \in \mathbb{R}^N$ 는 전이행렬 $\boldsymbol{\overline{A}}$ 를 가진 *은닉상태* 로 간주할 수 있습니다.
# $$
# \begin{aligned}
#   x_{k} &= \boldsymbol{\overline{A}} x_{k-1} + \boldsymbol{\overline{B}} u_k\\
#   y_k &= \boldsymbol{\overline{C}} x_k \\
# \end{aligned}
# $$

# 논문에서 언급하듯이, 이 "단계" 함수는 겉보기에
# RNN의 그것과 유사해 보입니다. 이를
# JAX의
# [scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)
# 함수를 사용하여 구현할 수 있습니다,


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


# 모든 것을 종합하여, 우리는 SSM을
# 먼저 이산화한 다음, 단계별로 반복함으로써 실행할 수 있습니다,

def run_SSM(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)

    # Run recurrence
    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]


# ### Tangent: A Mechanics Example

# SSM 구현을 더 직관적으로 이해하기 위해, 머신러닝에서 잠깐 물러서서, [역학분야에서의 고전적인 예제](https://en.wikipedia.org/wiki/State-space_representation#Moving_object_example)를 살펴봅니다.
 
# 이 예제에서는 한 덩어리가 벽으로부터 전방위치 $y(t)$ 에 스프링으로 연결되어 있습니다.
# 시간이 지나면서 이 덩어리는 다양한 힘 $u(t)$ 를 받습니다. 이 시스템의 매개변수는 질량 ($m$), 스프링 상수 ($k$), 마찰상수 ($b$) 로 구성되어 있습니다. 
# 다음의 미분방정식을 통해 이들 관계를 나타냅니다:

# $$\begin{aligned}
# my''(t) = u(t) - by'(t) - ky(t)
# \end{aligned}
# $$


# 행렬을 이용하면 다음의 요소로 이루어진 SSM 이 됩니다:

# $$
# \begin{aligned}
# \boldsymbol{A} &= \begin{bmatrix} 0 & 1 \\ -k/m & -b/m \end{bmatrix}  \\
# \boldsymbol{B} & = \begin{bmatrix} 0  \\ 1/m \end{bmatrix} & \boldsymbol{C} = \begin{bmatrix} 1 & 0  \end{bmatrix}  \\
# \end{aligned}
# $$


def example_mass(k, b, m):
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1.0 / m]])
    C = np.array([[1.0, 0]])
    return A, B, C


#  $\boldsymbol{C}$ 를 보면, 은닉상태의 첫번째 차원이 위치라는 것을 알 수 있습니다 ($y(t)$ 가 되기 때문).
#  두번째 차원은, $\boldsymbol{B}$ 를 통해 $u(t)$ 의 영향을 받으므로 속도가 됩니다. 
#  전이행렬, $\boldsymbol{A}$ 는 이러한 항들을 관련짓습니다.
#

# $u$ 를 $t$ 의 함수로 설정하면,


@partial(np.vectorize, signature="()->()")
def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)


#  이 코드로 SSM 을 실행합니다.


def example_ssm():
    # SSM
    ssm = example_mass(k=40, b=5, m=1)

    # L samples of u(t).
    L = 100
    step = 1.0 / L
    ks = np.arange(L)
    u = example_force(ks * step)

    # Approximation of y(t).
    y = run_SSM(*ssm, u)

    # Plotting ---
    import matplotlib.pyplot as plt
    import seaborn
    from celluloid import Camera

    seaborn.set_context("paper")
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    camera = Camera(fig)
    ax1.set_title("Force $u_k$")
    ax2.set_title("Position $y_k$")
    ax3.set_title("Object")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])

    # Animate plot over time
    for k in range(0, L, 2):
        ax1.plot(ks[:k], u[:k], color="red")
        ax2.plot(ks[:k], y[:k], color="blue")
        ax3.boxplot(
            [[y[k, 0] - 0.04, y[k, 0], y[k, 0] + 0.04]],
            showcaps=False,
            whis=False,
            vert=False,
            widths=10,
        )
        camera.snap()
    anim = camera.animate()
    anim.save("images/line.gif", dpi=150, writer="imagemagick")


if False:
    example_ssm()

# <img src="images/line.gif" width="100%">

# 멋지네요! 은닉상태가 2개에 불과한 SSM 하나이며, 100 단계에 걸쳐 있습니다.
# 최종모델은 **수천 스텝** 에 걸친 **수백개 스택의 SSM** 이 될 것입니다. 하지만 우선 이 모델들을 실제로 훈련될 수 있도록 만들어야 합니다.

# ### SSM 훈련: The Convolutional Representation

# 이 섹션에서 중요한 점은 위의 "RNN" 을 언롤링을 통해 "CNN" 으로 변환시킬 수 있다는 점입니다. 유도를 해 봅시다.

# > recurrent SSM 은 시퀀셜한 속성때문에 현대 하드웨어를 이용하여 훈련하기에 실용적이지 않습니다.
# > 대신, linear time-invariant (LTI) SSMs 과 연속 컨볼루션 사이에 잘 알려진 관계가 있습니다.
# > 따라서 Recurrent SSM 은 사실 [discrete convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution) 으로 표현할 수 있습니다. 
# > 
# > 초기상태를 간단히 $x_{-1} = 0$ 라고 하면 명시적으로 펼치면 다음과 같이 됩니다:
# >
# $$
# \begin{aligned}
#   x_0 &= \boldsymbol{\overline{B}} u_0 &
#   x_1 &= \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{B}} u_1 &
#   x_2 &= \boldsymbol{\overline{A}}^2 \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_1 + \boldsymbol{\overline{B}} u_2 & \dots
#   \\
#   y_0 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_0 &
#   y_1 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_1 &
#   y_2 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^2 \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_1 + \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_2
#   & \dots
# \end{aligned}
# $$
# >
# > 컨볼루션 커널에 관한 명시적 공식으로 컨볼루션으로 벡터화될 수 있습니다.
# >
# $$
# \begin{aligned}
#     y_k &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^k \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^{k-1} \boldsymbol{\overline{B}} u_1 + \dots + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_{k-1} + \boldsymbol{\overline{C}}\boldsymbol{\overline{B}} u_k
#     \\
#     y &= \boldsymbol{\overline{K}} \ast u
# \end{aligned}
# $$
# >
# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}} \in \mathbb{R}^L  = (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
# \end{aligned}
# $$
# $\boldsymbol{\overline{K}}$ 을 **SSM convolution kernel** 이나 필터라고 부릅니다.
    
# *엄청난 크기의* 필터입니다. 전체 시퀀스 크기입니다!


def K_conv(Ab, Bb, Cb, L):
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


# Warning: this implementation is naive and unstable. In practice it will fail to work
# for more than very small lengths. However, we are going to replace it with S4 in Part 2, so for
# now we just keep it around as a placeholder.


# 이 필터를 적용한 결과는 표준디렉트컨볼루션을 사용하거나, [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Convolution_theorem) 을 사용한 컨볼루션 정리를 사용하여 계산할 수 있습니다.
# 이산 컨볼루션 정리는 두 시퀀스의 원형 컨볼루션을 효율적으로 계산할 수 있게 해주며, 입력 시퀀스의 FFT를 먼저 곱한 다음 역 FFT 를 적용하여 컨볼루션의 출력을 계산합니다. 
# 우리 사례와 같은 비원형 컨볼루션에 이 정리를 활용하기 위해서는, 입력시퀀스를 0으로 패딩한 다음 출력 시퀀스의 패딩을 제거해야 합니다. 
# 길이가 길어질수록 이 FFT 방법은 직접컨볼루션보다 더 효율적입니다.

def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


# CNN 방법과 RNN 방법은 (대략) 같은 결과를 냅니다,


def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):
    ssm = random_SSM(rng, N)
    u = jax.random.uniform(rng, (L,))
    jax.random.split(rng, 3)
    # RNN
    rec = run_SSM(*ssm, u)

    # CNN
    ssmb = discretize(*ssm, step=step)
    conv = causal_convolution(u, K_conv(*ssmb, L))

    # Check
    assert np.allclose(rec.ravel(), conv.ravel())


# ### An SSM Neural Network.

# We now have all of the machinery needed to build a basic SSM neural network layer.
# As defined above, the discrete SSM defines a map from $\mathbb{R}^L
# \to \mathbb{R}^L$, i.e. a 1-D sequence map. We assume that we
# are going to be learning the parameters $B$ and $C$, as well as a
# step size $\Delta$ and a scalar $D$ parameter. The HiPPO matrix is
# used for the transition $A$. We learn the step size in log space.


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


# For the SSM layer most of the work is to build the filter.
# The actual call to the network is just the (huge) convolution we specified above.
#
# Note for Torch users: `setup` in Flax is called each time the parameters are updated.
# This is similar to the
# [Torch parameterizations](https://pytorch.org/tutorials/intermediate/parametrizations.html).
#
# As noted above this same layer can be used either as an RNN or a CNN. The argument
# `decode` determines which path is used. In the case of RNN we cache the previous state
# at each call in a Flax variable collection called `cache`.


class SSMLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


# Since our SSMs operate on scalars, we make $H$ different, stacked copies ($H$ different SSMs!) with
# different parameters. Here we use the [Flax vmap](
# https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.vmap.html)
# method to easily define these copies,


def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


SSMLayer = cloneLayer(SSMLayer)


# This SSM Layer can then be put into a standard NN.
# Here we add a block that pairs a call to an SSM with
# dropout and a linear projection.


class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x


# We can then stack a bunch of these blocks on top of each other
# to produce a stack of SSM layers. This can be used for
# classification or generation in the standard way as a Transformer.


class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Extra arguments to pass into layer constructor
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False  # Probably should be moved into layer_args

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0  # Normalize
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


# In Flax we add the batch dimension as a lifted transformation.
# We need to route through several variable collections which
# handle RNN and parameter caching (described below).


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)


# Overall, this defines a sequence-to-sequence map of shape (batch size, sequence length, hidden dimension),
# exactly the signature exposed by related sequence models such as Transformers, RNNs, and CNNs.

# Full code for training is defined in
# [training.py](https://github.com/srush/s4/blob/main/s4/train.py).

# 메인모델을 만들었지만, *SSMs 에 두 가지 핵심 문제* 가 있습니다. 
# 첫번째로, 랜덤으로 초기화된 SSM 은 실제로 잘 작동하지 않습니다. 더군다나, 지금까지 한 것처럼 순진하게 계산하면 매우 느리고, 메모리 비효율적이 됩니다.
# 다음으로, long-range dependencies 를 위해 특별한 초기화를 정의해서 S4 의 모델링 측면에 대해 논의를 완성하려고 합니다. 그 후 이 SSM 을 훨씬 빠르게 (<a href="#part-2-implementing-s4">Part 2</a>) 하는 법을 알아봅니다!

# ## Part 1b: HiPPO 로 Long-Range Dependencies 해결

# <img src="images/hippo.png" width="100%"/>
#
# > [이전 연구](https://arxiv.org/abs/2008.07669) 에서 기본 SSM 은 실제로 성능이 매우 나빴습니다. 직관적인 설명은,  시퀀스 길이에서 기하급수적으로 그래디언트 스케일링 문제(i.e., the
# > vanishing/exploding gradients problem) 입니다. 이를 해결하기 위해 이전 연구에서 HiPPO theory of continuous-time memorization 이 개발되었습니다.
# >
# > HiPPO 는 특정 행렬 $\boldsymbol{A} \in \mathbb{R}^{N \times N}$ 의 클래스를 지정합니다. 
# > 이 행렬이 포함되었을 때, 상태 $x(t)$ 가 입력 $u(t)$ 의 과거를 기억할 수 있도록 합니다.
# > 이 클래스에서 가장 중요한 행렬은 HiPPO 행렬로 정의됩니다.
# $$
# \begin{aligned}
#   (\text{\textbf{HiPPO Matrix}})
#   \qquad
#   \boldsymbol{A}_{nk}
#   =
#   \begin{cases}
#     (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\
#     n+1 & \text{if } n = k \\
#     0 & \text{if } n < k
#   \end{cases}
# \end{aligned}
# $$
# >
# > 이전 연구에서는 단순히 무작위 행렬 $\boldsymbol{A}$ 에서 HiPPO 로 SSM 을 변경하는 것이 시퀀셜 MNIST 분류 벤치마크에서 성능을 $60\%$ 에서 $98\%$ 로 향상시켰습니다.


# 이 행렬은 매우 중요할 것이지만, 마법처럼 보입니다. 우리의 목적을 위해서는 주로 1) 이 행렬을 한 번만 계산하면 된다는 것과 2) 이 행렬이 좋고, 단순한 구조를 가지고 있다는 것(우리는 이를 2부에서 활용할 것입니다)을 알아야 합니다. ODE 수학에 깊이 들어가지 않아도, 이 행렬의 주된 목표는 과거의 역사를 압축하여 역사를 대략적으로 재구성할 수 있는 충분한 정보를 가진 상태로 만드는 것입니다.

def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A

# 조금 더 깊이 들어가 보면, 이 행렬의 직관적인 설명은 그것이 그 역사를 기억하는 숨겨진 상태를 만들어낸다는 것입니다. 이는 [Legendre polynomial](https://en.wikipedia.org/wiki/Legendre_polynomials)의 계수를 추적함으로써 이루어집니다. 이 계수들은 그것이 이전의 모든 역사를 근사하게 할 수 있게 합니다. 예를 들어 살펴보겠습니다,

def example_legendre(N=8):
    # Random hidden state as coefficients
    import numpy as np
    import numpy.polynomial.legendre

    x = (np.random.rand(N) - 0.5) * 2
    t = np.linspace(-1, 1, 100)
    f = numpy.polynomial.legendre.Legendre(x)(t)

    # Plot
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_context("talk")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection="3d")
    ax.plot(
        np.linspace(-25, (N - 1) * 100 + 25, 100),
        [0] * 100,
        zs=-1,
        zdir="x",
        color="black",
    )
    ax.plot(t, f, zs=N * 100, zdir="y", c="r")
    for i in range(N):
        coef = [0] * N
        coef[N - i - 1] = 1
        ax.set_zlim(-4, 4)
        ax.set_yticks([])
        ax.set_zticks([])
        # Plot basis function.
        f = numpy.polynomial.legendre.Legendre(coef)(t)
        ax.bar(
            [100 * i],
            [x[i]],
            zs=-1,
            zdir="x",
            label="x%d" % i,
            color="brown",
            fill=False,
            width=50,
        )
        ax.plot(t, f, zs=100 * i, zdir="y", c="b", alpha=0.5)
    ax.view_init(elev=40.0, azim=-45)
    fig.savefig("images/leg.png")


if False:
    example_legendre()

# 빨간 선은 우리가 근사하고 있는 그 곡선을 나타내며, 검은 막대는 우리의 숨겨진 상태의 값들을 나타냅니다. 각각은 파란색 함수로 나타낸 르장드르 급수의 한 요소에 대한 계수입니다. 직관적으로 이해하자면, HiPPO 행렬은 이러한 계수들을 각 단계마다 업데이트합니다.

# <img src="images/leg.png" width="100%">


# ## Part 2: S4 구현

# Warning: this section has a lot of math. Roughly it boils down to finding a
# way to compute the filter from Part 1 for "HiPPO-like" matrices *really
# fast*. If you are interested, the details are really neat. If not,
# skip to Part 3 for some cool applications like MNIST completion.

# [Skip Button](#part-3-s4-in-practice)

# S4가 기본 SSM과 두 가지 주요한 차이점을 가지고 있다는 것을 기억하세요. 첫 번째는 이전 부분에서 정의된 $\boldsymbol{A}$ 행렬에 대한 특별한 공식을 사용함으로써 *모델링에서의 문제* 즉, 장거리 의존성 - 을 해결합니다. 이러한 특별한 SSM은 [선행](https://arxiv.org/abs/2110.13985) 연구들에서 S4 에 고려되었습니다.
    
# The second main feature of S4 solves the *computational challenge* of SSMs by introducing a special representation and algorithm to be able to work with this matrix!


# > The fundamental bottleneck in computing the discrete-time SSM
# > is that it involves repeated matrix multiplication by
# > $\boldsymbol{\overline{A}}$.  For example, computing
# > naively  involves $L$ successive multiplications
# > by $\boldsymbol{\overline{A}}$, requiring $O(N^2 L)$ operations and
# > $O(NL)$ space.

# Specifically, recall this function here:

# ```python
# def K_conv(Ab, Bb, Cb, L):
#    return np.array(
#        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
#    )
# ```

# The contribution of S4 is a stable method for speeding up this particular operation.
# To do this we are going to focus on the case where the SSM
# has special structure: specifically, Diagonal Plus Low-Rank (DPLR) in complex
# space.

# A **DPLR** SSM is $(\boldsymbol{\Lambda} - \boldsymbol{P}\boldsymbol{Q}^*, \boldsymbol{B}, \boldsymbol{C})$ for some diagonal $\boldsymbol{\Lambda}$ and matrices $\boldsymbol{P}, \boldsymbol{Q}, \boldsymbol{B}, \boldsymbol{C} \in \mathbb{C}^{N \times 1}$.
# We assume without loss of generality that the rank is $1$, i.e. these matrices are vectors.
#
# Under this DPLR assumption, S4 overcomes the speed bottleneck in three steps

#
# >  1. Instead of computing $\boldsymbol{\overline{K}}$ directly,
#     we compute its spectrum by evaluating its **[truncated generating function](https://en.wikipedia.org/wiki/Generating_function)** .  This  now involves a matrix *inverse* instead of *power*.
# >  2. We show that the diagonal matrix case is equivalent to the computation of a **[Cauchy kernel](https://en.wikipedia.org/wiki/Cauchy_matrix)** $\frac{1}{\omega_j - \zeta_k}$.
# >  3. We show the low-rank term can now be corrected by applying the **[Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)** which reduces $(\boldsymbol{\Lambda} + \boldsymbol{P}\boldsymbol{Q}^*)^{-1}$ in terms of $\boldsymbol{\Lambda}^{-1}$, truly reducing to the diagonal case.


# ### Step 1. SSM Generating Functions

# The main step will be switching from computing the sequence to computing its generating function.
# From the paper's appendix:

# > To address the problem of computing powers of $\boldsymbol{\overline{A}}$, we introduce another technique.
# > Instead of computing the SSM convolution filter $\boldsymbol{\overline{K}}$ directly,
# > we introduce a generating function on its coefficients and compute evaluations of it.
# >
# >The *truncated SSM generating function* at node $z$ with truncation $L$ is
# $$
# \hat{\mathcal{K}}_L(z; \boldsymbol{\overline{A}}, \boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}) \in \mathbb{C} := \sum_{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i
# $$


def K_gen_simple(Ab, Bb, Cb, L):
    K = K_conv(Ab, Bb, Cb, L)

    def gen(z):
        return np.sum(K * (z ** np.arange(L)))

    return gen


# > The generating function essentially converts the SSM convolution filter from the time domain to
# > frequency domain. This transformation is also called [z-transform](https://en.wikipedia.org/wiki/Z-transform) (up to a minus sign) in control engineering literature. Importantly, it preserves the same information, and the desired SSM convolution filter
# > can be recovered. Once the z-transform of a discrete sequence known, we can obtain the filter's discrete fourier transform from evaluations of its
# > [z-transform at the roots of unity](https://en.wikipedia.org/wiki/Z-transform#Inverse_Z-transform)
# $\Omega = \{ \exp(2\pi \frac{k}{L} : k \in [L] \}$. Then, we can apply inverse fourier transformation, stably in $O(L \log L)$ operations by applying an [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform), to recover the filter.


def conv_from_gen(gen, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))
    atRoots = jax.vmap(gen)(Omega_L)
    # Inverse FFT
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real


# More importantly, in the generating function we can replace the matrix power with an inverse!
# $$
# \hat{\mathcal{K}}_L(z) = \sum_{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i = \boldsymbol{\overline{C}} (\boldsymbol{I} - \boldsymbol{\overline{A}}^L z^L) (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}} = \boldsymbol{\widetilde{C}}  (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}}
# $$

# And for all $z \in \Omega_L$, we have $z^L = 1$ so that term is removed. We then pull this constant
# term into a new $\boldsymbol{\widetilde{C}}$. Critically, this function **does not** call `K_conv`,


def K_gen_inverse(Ab, Bb, Cb, L):
    I = np.eye(Ab.shape[0])
    Ab_L = matrix_power(Ab, L)
    Ct = Cb @ (I - Ab_L)
    return lambda z: (Ct.conj() @ inv(I - Ab * z) @ Bb).reshape()


# But it does output the same values,


def test_gen_inverse(L=16, N=4):
    ssm = random_SSM(rng, N)
    ssm = discretize(*ssm, 1.0 / L)
    b = K_conv(*ssm, L=L)

    a = conv_from_gen(K_gen_inverse(*ssm, L=L), L)
    assert np.allclose(a, b)


#  In summary, Step 1 allows us to replace the matrix power with an
#  inverse by utilizing a truncated generating function.
#  However this inverse still needs to be calculated $L$
#  times (for each of the roots of unity).

# ### Step 2: Diagonal Case

# The next step to assume special *structure* on the matrix
# $\boldsymbol{A}$ to compute the inverse faster than the naive inversion.
# To begin, let us first convert the equation above to use the original SSM
# matrices. With some algebra you can expand the discretization and show:

# $$
# \begin{aligned}
#   \boldsymbol{\widetilde{C}}\left(\boldsymbol{I} - \boldsymbol{\overline{A}} \right)^{-1} \boldsymbol{\overline{B}}
#   =
#   \frac{2\Delta}{1+z} \boldsymbol{\widetilde{C}} \left[ {2 \frac{1-z}{1+z}} - \Delta \boldsymbol{A} \right]^{-1} \boldsymbol{B}
# \end{aligned}
# $$


# Now imagine $\boldsymbol{A}=\boldsymbol{\Lambda}$ for a diagonal $\boldsymbol{\Lambda}$. Substituting in the discretization
# formula the authors show that the generating function can be written in the following manner:

# $$ \begin{aligned}
# \boldsymbol{\hat{K}}_{\boldsymbol{\Lambda}}(z) & = c(z) \sum_i \cdot \frac{\boldsymbol{\widetilde{C}}_i \boldsymbol{B}_i} {(g(z) - \boldsymbol{\Lambda}_i)} = c(z) \cdot k_{z, \boldsymbol{\Lambda}}(\boldsymbol{\widetilde{C}}, \boldsymbol{B}) \\
#  \end{aligned}$$
# where $c$ is a constant, and $g$ is a function of $z$.


# We have effectively replaced an inverse with a weighted dot product.
# Let's make a small helper function to compute this weight dot product for use.


def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()


# While not important for our implementation, it is worth noting that
# this is a [Cauchy
# kernel](https://en.wikipedia.org/wiki/Cauchy_matrix) and is the
# subject of many other [fast
# implementations](https://en.wikipedia.org/wiki/Fast_multipole_method).


# ### Step 3: Diagonal Plus Low-Rank

# The final step is to relax the diagonal assumption. In addition to
# the diagonal term we allow a low-rank component with
# $\boldsymbol{P}, \boldsymbol{Q} \in \mathbb{C}^{N\times 1}$ such that:

# $$
# \boldsymbol{A} = \boldsymbol{\Lambda} - \boldsymbol{P}  \boldsymbol{Q}^*
# $$

# The [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
# tells us that the inverse of a diagonal plus rank-1 term is equal to the
# inverse of the diagonal plus a rank-1 term. We write it out here
# adding the low-rank term.

# $$ \begin{aligned}
# (\boldsymbol{\Lambda} + \boldsymbol{P}  \boldsymbol{Q}^*)^{-1} &= \boldsymbol{\Lambda}^{-1} - \boldsymbol{\Lambda}^{-1} \boldsymbol{P} (1 + \boldsymbol{Q}^* \boldsymbol{\Lambda}^{-1} \boldsymbol{P})^{-1} \boldsymbol{Q}^* \boldsymbol{\Lambda}^{-1}
#  \end{aligned}
# $$

#  There is a bunch of algebra in the appendix. It mostly consists of substituting this component in for A,
#  applying the Woodbury identity and distributing terms. We end up with 4 terms that
#  all look like Step 2 above:

# $$ \begin{aligned}
# \boldsymbol{\hat{K}}_{DPLR}(z) & = c(z) [k_{z, \Lambda}(\boldsymbol{\widetilde{C}}, \boldsymbol{\boldsymbol{B}}) - k_{z, \Lambda}(\boldsymbol{\widetilde{C}}, \boldsymbol{\boldsymbol{P}}) (1 + k_{z, \Lambda}(\boldsymbol{q^*}, \boldsymbol{\boldsymbol{P}}) )^{-1} k_{z, \Lambda}(\boldsymbol{q^*}, \boldsymbol{\boldsymbol{B}}) ]
#  \end{aligned}$$


# The code consists of collecting up the terms and applying 4 weighted dot products,


def K_gen_DPLR(Lambda, P, Q, B, C, step, unmat=False):
    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    def gen(o):
        g = (2.0 / step) * ((1.0 - o) / (1.0 + o))
        c = 2.0 / (1.0 + o)

        def k(a):
            # Checkpoint this calculation for memory efficiency.
            if unmat:
                return jax.remat(cauchy_dot)(a, g, Lambda)
            else:
                return cauchy_dot(a, g, Lambda)

        k00 = k(aterm[0] * bterm[0])
        k01 = k(aterm[0] * bterm[1])
        k10 = k(aterm[1] * bterm[0])
        k11 = k(aterm[1] * bterm[1])
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


# This is our final version of the $K$ function. Because `conv_from_gen` is always called together with a generating function (e.g. `K_gen_DPLR`), we'll fuse them into define a dedicated function to compute the DPLR SSM kernel from all of its parameters. (With fewer layers of indirection, this could also make it easier for XLA compiler to optimize.)


@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real


# Now we can check whether it worked.
# First, let's generate a random Diagonal Plus Low Rank (DPLR) matrix,


def random_DPLR(rng, N):
    l_r, p_r, q_r, b_r, c_r = jax.random.split(rng, 5)
    Lambda = jax.random.uniform(l_r, (N,))
    P = jax.random.uniform(p_r, (N,))
    Q = jax.random.uniform(q_r, (N,))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return Lambda, P, Q, B, C


# We can check that the DPLR method yields the same filter as computing $\boldsymbol{A}$ directly,


def test_gen_dplr(L=16, N=4):
    I = np.eye(4)

    # Create a DPLR A matrix and discretize
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    A = np.diag(Lambda) - P[:, np.newaxis] @ P[:, np.newaxis].conj().T
    _, _, C = random_SSM(rng, N)

    Ab, Bb, Cb = discretize(A, B, C, 1.0 / L)
    a = K_conv(Ab, Bb, Cb.conj(), L=L)

    # Compare to the DPLR generating function approach.
    C = (I - matrix_power(Ab, L)).conj().T @ Cb.ravel()
    b = kernel_DPLR(Lambda, P, P, B, C, step=1.0 / L, L=L)
    assert np.allclose(a.real, b.real)


# ### Diagonal Plus Low-Rank RNN.

# A secondary benefit of the DPLR factorization is that it allows
# us to compute the discretized form of the SSM without having
# to invert the $A$ matrix directly. Here we return to the paper
# for the derivation.

# > Recall that discretization computes,
# $$
# \begin{align*}
#   \bm{\overline{A}} &= (\bm{I} - \Delta/2 \cdot \bm{A})^{-1}(\bm{I} + \Delta/2 \cdot \bm{A}) \\
#   \bm{\overline{B}} &= (\bm{I} - \Delta/2 \cdot \bm{A})^{-1} \Delta \bm{B}
#   .
# \end{align*}
# $$
# >
# > We simplify both terms in the definition of $\bm{\overline{A}}$ independently.
# > The first term is:
# $$
# \begin{align*}
#   \bm{I} + \frac{\Delta}{2} \bm{A}
#   &= \bm{I} + \frac{\Delta}{2} (\bm{\Lambda} - \bm{P} \bm{Q}^*)
#   \\&= \frac{\Delta}{2} \left[ \frac{2}{\Delta}\bm{I} + (\bm{\Lambda} - \bm{P} \bm{Q}^*) \right]
#   \\&= \frac{\Delta}{2} \bm{A_0}
# \end{align*}
# $$
# > where $\bm{A_0}$ is defined as the term in the final brackets.
# >
# > The second term is known as the Backward Euler's method.
# > Although this inverse term is normally difficult to deal with,
# > in the DPLR case we can simplify it using Woodbury's Identity as described above.
# $$
# \begin{align*}
#   \left( \bm{I} - \frac{\Delta}{2} \bm{A} \right)^{-1}
#   &=
#   \left( \bm{I} - \frac{\Delta}{2} (\bm{\Lambda} - \bm{P} \bm{Q}^*) \right)^{-1}
#   \\&=
#   \frac{2}{\Delta} \left[ \frac{2}{\Delta} - \bm{\Lambda} + \bm{P} \bm{Q}^* \right]^{-1}
#   \\&=
#   \frac{2}{\Delta} \left[ \bm{D} - \bm{D} \bm{P} \left( 1 + \bm{Q}^* \bm{D} \bm{P} \right)^{-1} \bm{Q}^* \bm{D} \right]
#   \\&= \frac{2}{\Delta} \bm{A_1}
# \end{align*}
# $$
# > where $\bm{D} = \left( \frac{2}{\Delta}-\bm{\Lambda} \right)^{-1}$
# > and $\bm{A_1}$ is defined as the term in the final brackets.
# >
# >  The discrete-time SSM \eqref{eq:2} becomes
# $$
# \begin{align*}
#   x_{k} &= \bm{\overline{A}} x_{k-1} + \bm{\overline{B}} u_k \\
#   &= \bm{A_1} \bm{A_0} x_{k-1} + 2 \bm{A_1} \bm{B} u_k \\
#   y_k &= \bm{C} x_k
#   .
# \end{align*}
# $$


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


# ### Turning HiPPO to DPLR

# This approach applies to DPLR matrices, but remember we would like it to also apply to the HiPPO matrix.
#  While not DPLR in its current form, the HiPPO matrix *does have special structure*. It is
#  Normal Plus Low-Rank (NPLR). Because [normal](https://en.wikipedia.org/wiki/Normal_matrix) matrices are exactly the class of matrices that are unitarily diagonalizable, NPLR matrices are essentially equivalent to DPLR matrices from the perspective of SSM models.
# this is just as good as DPLR for the purposes of learning an SSM network.

# > The S4 techniques can apply to any matrix $\boldsymbol{A}$ that can be decomposed as *Normal Plus Low-Rank (NPLR)*.
# $$
# >   \boldsymbol{A} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V}^* - \boldsymbol{P} \boldsymbol{Q}^\top = \boldsymbol{V} \left( \boldsymbol{\Lambda} - \boldsymbol{V}^* \boldsymbol{P} (\boldsymbol{V}^*\boldsymbol{Q})^* \right) \boldsymbol{V}^*
# $$
# > for [unitary](https://en.wikipedia.org/wiki/Unitary_matrix) $\boldsymbol{V} \in \mathbb{C}^{N \times N}$, diagonal $\boldsymbol{\Lambda}$, and low-rank factorization $\boldsymbol{P}, \boldsymbol{Q} \in \mathbb{R}^{N \times r}$.  An NPLR SSM is therefore unitarily equivalent to some DPLR matrix.


#  For S4, we need to work with a HiPPO matrix for $\boldsymbol{A}$. This requires first writing it as a normal plus low-rank term, and then diagonalizing to extract
#  $\boldsymbol{\Lambda}$ from this decomposition. The appendix of the paper shows how
#  by writing the normal part as a [skew-symmetric](https://en.wikipedia.org/wiki/Skew-symmetric_matrix) (plus a constant times the identity matrix), which are a special class of normal matrices.

# An additional simplification is that there is actually a representation that ties the low-rank components terms $\boldsymbol{P} = \boldsymbol{Q}$, which was shown in [follow-up work](https://arxiv.org/abs/2202.09729) to be important for stability.


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B


# After extracting the normal part, we can diagonalize to get out the DPLR terms.
# Because the normal part is actually skew-symmetric, we can extract the real and complex parts of $\boldsymbol{\Lambda}$ separately.
# This serves two purposes. First, this gives us finer-grained control over the real and imaginary parts, which can be used to improve stability. Second, this lets us use more powerful diagonalization algorithms for [Hermitian matrices](https://en.wikipedia.org/wiki/Hermitian_matrix) - in fact, the current version of JAX does not support GPU diagonalization for non-Hermitian matrices!


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


# Sanity check just to make sure those identities hold,


def test_nplr(N=8):
    A2, P, B = make_NPLR_HiPPO(N)
    Lambda, Pc, Bc, V = make_DPLR_HiPPO(N)
    Vc = V.conj().T
    P = P[:, np.newaxis]
    Pc = Pc[:, np.newaxis]
    Lambda = np.diag(Lambda)

    A3 = V @ Lambda @ Vc - (P @ P.T)  # Test NPLR
    A4 = V @ (Lambda - Pc @ Pc.conj().T) @ Vc  # Test DPLR
    assert np.allclose(A2, A3, atol=1e-4, rtol=1e-4)
    assert np.allclose(A2, A4, atol=1e-4, rtol=1e-4)


# ### Final Check

# This tests that everything works as planned.


def test_conversion(N=8, L=16):
    step = 1.0 / L
    # Compute a HiPPO NPLR matrix.
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    # Random complex Ct
    C = normal(dtype=np.complex64)(rng, (N,))

    # CNN form.
    K = kernel_DPLR(Lambda, P, P, B, C, step, L)

    # RNN form.
    Ab, Bb, Cb = discrete_DPLR(Lambda, P, P, B, C, step, L)
    K2 = K_conv(Ab, Bb, Cb, L=L)
    assert np.allclose(K.real, K2.real, atol=1e-5, rtol=1e-5)

    # Apply CNN
    u = np.arange(L) * 1.0
    y1 = causal_convolution(u, K.real)

    # Apply RNN
    _, y2 = scan_SSM(
        Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)).astype(np.complex64)
    )
    assert np.allclose(y1, y2.reshape(-1).real, atol=1e-4, rtol=1e-4)


# ## Part 3: S4 in Practice

# That was a lot of work, but now the actual model is concise. In fact
# we are only using four functions:


# 1. `K_gen_DPLR` → Truncated generating function when $\boldsymbol{A}$ is DPLR (S4-part)
# 2. `conv_from_gen` → Convert generating function to filter
# 3. `causal_convolution` → Run convolution
# 4. `discretize_DPLR` → Convert SSM to discrete form for RNN.


# ### S4 CNN / RNN Layer

#  A full S4 Layer is very similar to the simple SSM layer above. The
#  only difference is in the the computation of $\boldsymbol{K}$.
#  Additionally instead of learning $\boldsymbol{C}$, we learn
#  $\boldsymbol{\widetilde{C}}$ so we avoid computing powers of
#  $\boldsymbol{A}$. Note as well that in the original paper $\boldsymbol{\Lambda}, \boldsymbol{P}, \boldsymbol{Q}$ are
#  also learned. However, in this post, we leave them fixed for simplicity.


class S4Layer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


S4Layer = cloneLayer(S4Layer)

# We initialize the model by computing a HiPPO DPLR initializer


# Factory for constant initializer in Flax
def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)


# ### Sampling and Caching

# We can sample from the model using the RNN implementation. Here is
# what the sampling code looks like. Note that we keep a running cache
# to remember the RNN state.


def sample(model, params, prime, cache, x, start, end, rng):
    def loop(i, cur):
        x, rng, cache = cur
        r, rng = jax.random.split(rng)
        out, vars = model.apply(
            {"params": params, "prime": prime, "cache": cache},
            x[:, np.arange(1, 2) * i],
            mutable=["cache"],
        )

        def update(x, out):
            p = jax.random.categorical(r, out[0])
            x = x.at[i + 1, 0].set(p)
            return x

        x = jax.vmap(update)(x, out)
        return x, rng, vars["cache"].unfreeze()

    return jax.lax.fori_loop(start, end, jax.jit(loop), (x, rng, cache))[0]


# To get this in a good form, we first precompute the discretized
# version of the the RNN for each S4 layers. We do this through the
# "prime" collection of variables.


def init_recurrence(model, params, init_x, rng):
    variables = model.init(rng, init_x)
    vars = {
        "params": params,
        "cache": variables["cache"].unfreeze(),
        "prime": variables["prime"].unfreeze(),
    }
    print("[*] Priming")
    _, prime_vars = model.apply(vars, init_x, mutable=["prime"])
    return vars["params"], prime_vars["prime"], vars["cache"]


# Putting this altogether we can sample from a model directly.


def sample_checkpoint(path, model, length, rng):
    from flax.training import checkpoints

    start = np.zeros((1, length, 1), dtype=int)

    print("[*] Initializing from checkpoint %s" % path)
    state = checkpoints.restore_checkpoint(path, None)
    assert "params" in state
    params, prime, cache = init_recurrence(model, state["params"], start, rng)
    print("[*] Sampling output")
    return sample(model, params, prime, cache, start, 0, length - 1, rng)


# ### Experiments: MNIST

# Now that we have the model, we can try it out on some MNIST experiments.
# For these experiments we linearize MNIST and just treat each image as a sequence of
# pixels.

# The first experiments we ran were on MNIST classification. While
# not in theory a hard problem, treating MNIST as a linear sequence
# classification task is a bit strange. However in practice, the model
# with $H=256$ and four layers seems to get up near 99% right away.

# A more visually interesting task is generating MNIST digits, by predicting entire
# sequences of pixels! Here, we simply feed in a sequence of pixels into the model and have it
# predict the next one like language modeling. With a little
# tweaking, we are able to get the model to an NLL of 0.36 on this
# task with size 512 and 6 layers (~4m parameters).
#
# The metric usually used for this task is *[bits per
# dimension](https://paperswithcode.com/sota/image-generation-on-mnist)* which is
# NLL in base 2 for MNIST. A loss of 0.36 is ~0.52 BPD which is SOTA according to PapersWithCode.


# <img src="images/sample.png" width="100%">

# We can also do prefix-samples – given the first 300 pixels, try to complete the image.
# S4 is on the left, true on the right.

# <img src="images/im0.1.png" width="45%">
# <img src="images/im0.2.png" width="45%">
# <img src="images/im0.3.png" width="45%">
# <img src="images/im0.4.png" width="45%">
# <img src="images/im0.5.png" width="45%">
# <img src="images/im0.6.png" width="45%">
# <img src="images/im0.7.png" width="45%">
# <img src="images/im0.8.png" width="45%">


def sample_image_prefix(
    params,
    model,
    # length,
    rng,
    dataloader,
    prefix=300,
    # bsz=32,
    imshape=(28, 28),
    n_batches=None,
    save=True,
):
    """Sample a grayscale image represented as intensities in [0, 255]"""
    import matplotlib.pyplot as plt
    import numpy as onp

    # from .data import Datasets
    # BATCH = bsz
    # start = np.zeros((BATCH, length), dtype=int)
    # start = np.zeros((BATCH, length, 1), dtype=int)
    start = np.array(next(iter(dataloader))[0].numpy())
    start = np.zeros_like(start)
    # params, prime, cache = init_recurrence(model, params, start[:, :-1], rng)
    params, prime, cache = init_recurrence(model, params, start, rng)

    BATCH = start.shape[0]
    START = prefix
    LENGTH = start.shape[1]
    assert LENGTH == onp.prod(imshape)

    # _, dataloader, _, _, _ = Datasets["mnist"](bsz=BATCH)
    it = iter(dataloader)
    for j, im in enumerate(it):
        if n_batches is not None and j >= n_batches:
            break

        image = im[0].numpy()
        image = np.pad(
            image[:, :-1, :], [(0, 0), (1, 0), (0, 0)], constant_values=0
        )
        cur = onp.array(image)
        # cur[:, START + 1 :, 0] = 0
        # cur = np.pad(cur[:, :-1, 0], [(0, 0), (1, 0)], constant_values=256)
        cur = np.array(cur[:, :])

        # Cache the first `start` inputs.
        out, vars = model.apply(
            {"params": params, "prime": prime, "cache": cache},
            cur[:, np.arange(0, START)],
            mutable=["cache"],
        )
        cache = vars["cache"].unfreeze()
        out = sample(model, params, prime, cache, cur, START, LENGTH - 1, rng)

        # Visualization
        out = out.reshape(BATCH, *imshape)
        final = onp.zeros((BATCH, *imshape, 3))
        final2 = onp.zeros((BATCH, *imshape, 3))
        final[:, :, :, 0] = out
        f = final.reshape(BATCH, LENGTH, 3)
        i = image.reshape(BATCH, LENGTH)
        f[:, :START, 1] = i[:, :START]
        f[:, :START, 2] = i[:, :START]
        f = final2.reshape(BATCH, LENGTH, 3)
        f[:, :, 1] = i
        f[:, :START, 0] = i[:, :START]
        f[:, :START, 2] = i[:, :START]
        if save:
            for k in range(BATCH):
                fig, (ax1, ax2) = plt.subplots(ncols=2)
                ax1.set_title("Sampled")
                ax1.imshow(final[k] / 256.0)
                ax2.set_title("True")
                ax1.axis("off")
                ax2.axis("off")
                ax2.imshow(final2[k] / 256.0)
                fig.savefig("im%d.%d.png" % (j, k))
                plt.close()
                print(f"Sampled batch {j} image {k}")
    return final, final2


# ### Experiments: QuickDraw


# Next we tried training a model to generate drawings. For this we
# used the [QuickDraw
# dataset](https://github.com/googlecreativelab/quickdraw-dataset).
# The dataset includes a version of the dataset downsampled to MNIST
# size so we can use roughly the same model as above. The dataset
# is much larger though (5M images) and more complex. We only trained
# for 1 epoch with a $H=256$, 4 layer model. Still, the approach was
# able to generate relatively coherent completions. These are prefix
# samples with 500 pixels given.

# <img src="images/quickdraw/im1.png" width="45%">
# <img src="images/quickdraw/im2.png" width="45%">
# <img src="images/quickdraw/im3.png" width="45%">
# <img src="images/quickdraw/im4.png" width="45%">
# <img src="images/quickdraw/im5.png" width="45%">
# <img src="images/quickdraw/im6.png" width="45%">


# ### Experiments: Spoken Digits

# Finally we played with modeling sound waves directly. For these, we
# use the
# [Free Spoken Digits Datasets](https://github.com/Jakobovski/free-spoken-digit-dataset)
# an MNIST like dataset of various speakers reading off digits. We
# first trained a classification model and found that the approach was
# able to reach $97\%$ accuracy just from the raw soundwave. Next we
# trained a generation model to produce the sound wave directly. With
# $H=512$ the model seems to pick up the data relatively well. This
# dataset only has around 3000 examples, but the model can produce
# reasonably good (cherry-picked) continuations. Note these sequences are 6400 steps
# long at an 8kHz sampling rate, discretized to 256 classes with
# [Mu Law Encoding](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm).

# <center>
# <img src='images/speech3.1.png' width='100%'>
# <audio controls>
#  <source src='images/sample3.1.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample3.1.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech6.1.png' width='100%'>
# <audio controls>
#  <source src='images/sample6.1.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample6.1.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech7.0.png' width='100%'>
# <audio controls>
#  <source src='images/sample7.0.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample7.0.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech9.0.png' width='100%'>
# <audio controls>
#  <source src='images/sample9.0.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample9.0.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech10.0.png' width='100%'>
# <audio controls>
#  <source src='images/sample10.0.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample10.0.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech13.1.png' width='100%'>
# <audio controls>
#  <source src='images/sample13.1.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample13.1.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech23.0.png' width='100%'>
# <audio controls>
#  <source src='images/sample23.0.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample23.0.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech25.0.png' width='100%'>
# <audio controls>
#  <source src='images/sample25.0.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample25.0.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech26.0.png' width='100%'>
# <audio controls>
#  <source src='images/sample26.0.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample26.0.gold.wav' type='audio/wav'>
# </audio>
#
# <img src='images/speech26.1.png' width='100%'>
# <audio controls>
#  <source src='images/sample26.1.wav' type='audio/wav'>
# </audio>
# <audio controls>
#  <source src='images/sample26.1.gold.wav' type='audio/wav'>
# </audio>
# </center>

# Our [full code base](https://github.com/srush/annotated-s4/) contains
# more examples and infrastructure for training models for generations and
# classification.

# ## Conclusion


# Putting together this post inspired lots of thoughts about future
# work in this area. One obvious conclusion is that long-range
# models have all sorts of future applications from acoustic modeling to
# genomic sequences to trajectories (not to mention our shared area of
# NLP). Another is some surprise that linear models can be so effective
# here, while also opening up a range of efficient techniques.
# Finally from a practical level, the transformations in JAX
# make it really nice to implement complex models like this
# in a very concise way (~200 LoC), with similar efficiency and performance!

# We end by thanking the authors [Albert Gu](http://web.stanford.edu/~albertgu/) and
# [Karan Goel](https://krandiash.github.io/), who were super helpful in
# putting this together, and pointing you again to their
# [paper](https://arxiv.org/abs/2111.00396) and
# [codebase](https://github.com/HazyResearch/state-spaces).
# Thanks to Ankit Gupta, Ekin Akyürek, Qinsheng Zhang, Nathan Yan, and Junxiong Wang for
# contributions.
# We're also grateful for Conner Vercellino and
# Laurel Orr for providing helpful feedback on this post.

#
# / Cheers – Sasha & Sidd


# ## Changelog
#
# * v3
#   * Major editing pass from Albert.
#   * Fix bug in HiPPO calculation.
#   * Added training of all S4 parameters.
#   * Fix learning rate / initialization issues.
# * v2
#   * Added RNN decoding
#   * Added Speech examples
# * v1 - Original version
