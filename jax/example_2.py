import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

## JAX vs. NumPy

# numpy로 할 수 있는 거의 모든 것이 jax.numpy로도 할 수 있습니다:
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)
# plt.show()

print(f"{type(x_np)=}, {type(x_jnp)=}")

# JAX 배열과 NumPy 배열 사이에는 하나의 중요한 차이가 있습니다: 
# JAX 배열은 불변입니다, 
# 즉 생성된 후 그 내용을 변경할 수 없다는 의미입니다.

# NumPy: 가변 배열
x = np.arange(10)
x[0] = 10
print(x)

# JAX: 불변 배열 => 오류가 발생합니다:
x = jnp.arange(10)
# x[0] = 10

# 개별 요소를 업데이트하기 위해, 
# JAX는 업데이트된 복사본을 반환하는 인덱스 업데이트 문법을 제공합니다.
y = x.at[0].set(10)
print(x)
print(y)

## NumPy, lax & XLA: JAX API 계층화
import jax.numpy as jnp
from jax import lax

jnp.add(1, 1.0)  # jax.numpy API는 혼합 타입을 암시적으로 승격합니다.
# lax.add(1, 1.0)  # jax.lax API는 명시적인 타입 승격을 요구합니다.

# 타입 승격을 명시적으로 수행해야 합니다.
lax.add(jnp.float32(1), 1.0)

# jax.lax는 NumPy가 지원하는 것보다 더 일반적인 연산들에 대해 
# 효율적인 API를 제공합니다.
x = jnp.array([1, 2, 1])
y = jnp.ones(10)
jnp.convolve(x, y)

## JIT을 할지 말지
import time
from jax import jit

def norm(X):
    X = X - X.mean(0)
    return X / X.std(0)

# jax.jit 변환을 사용하여 함수의 즉시 컴파일된 버전을 만들 수 있습니다:
norm_compiled = jit(norm)

np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
print(np.allclose(norm(X), norm_compiled(X), atol=1E-6))

start = time.time()
y = norm(X).block_until_ready()
end = time.time()
print(f"[Numpy] Execution time: {end - start:.4f} seconds")

start = time.time()
y = norm_compiled(X).block_until_ready()
end = time.time()
print(f"[JIT] Execution time: {end - start:.4f} seconds")

# jax.jit에는 제한 사항이 있습니다. 
# 특히, 모든 배열이 정적인 형태를 가져야 한다는 것입니다. 
# 이는 일부 JAX 연산이 JIT 컴파일과 호환되지 않음을 의미합니다.

def get_negatives(x):
    return x[x < 0]

x = jnp.array(np.random.randn(10))
print(f"{get_negatives(x)=}")

# 이는 함수가 컴파일 시간에 알려지지 않은 형태의 배열을 생성하기 때문입니다: 
# 출력의 크기는 입력 배열의 값에 따라 달라지므로, JIT과 호환되지 않습니다.
# jit(get_negatives)(x)

# 이러한 제한을 극복하기 위해, 
# 함수를 두 단계로 나누어 컴파일할 수 있습니다:
def get_negatives_compiled(x):
    x = jnp.where(x < 0, x, 0)
    return x

print(f"{jit(get_negatives_compiled)(x)=}")

## JIT 메커니즘: 트레이싱과 정적 변수

@jit
def f(x, y):
    print("Running f():")
    print(f"  x = {x}")
    print(f"  y = {y}")
    result = jnp.dot(x + 1, y + 1)
    print(f"  result = {result}")
    return result

x = np.random.randn(3, 4)
y = np.random.randn(4)
start = time.time()
result = f(x, y).block_until_ready()
end = time.time()
print(f"[JIT] First Execution time: {end - start:.4f} seconds")

x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
start = time.time()
result = f(x2, y2).block_until_ready()
end = time.time()
print(f"[JIT] Second Execution time: {end - start:.4f} seconds")

# 추출된 연산 시퀀스는 JAX 표현식, 또는 짧게는 jaxpr에 인코딩됩니다.
# jax.make_jaxpr 변환을 사용하여 jaxpr을 볼 수 있습니다:
from jax import make_jaxpr

def f(x, y):
    return jnp.dot(x + 1, y + 1)

print(make_jaxpr(f)(x, y))

# 배열의 내용에 대한 정보 없이 JIT 컴파일이 수행되기 때문에, 
# 함수 내의 제어 흐름 문장은 추적된 값에 의존할 수 없습니다. 
# 예를 들어, 다음 코드는 오류를 발생시킵니다:

@jit
def f(x, neg):
    return -x if neg else x

# f(1, True)

# 추적하고 싶지 않은 변수가 있다면, JIT 컴파일을 위해 static으로 표시할 수 있습니다:
from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
    return -x if neg else x
print(f"{f(1, True)=}")
print(f"{f(1, False)=}")

## 정적 연산 vs 추적 연산
@jit
def f(x):
    print(f"x = {x}")
    print(f"x.shape = {x.shape}")
    print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
    # 이 오류를 피하기 위해 이 부분을 주석 처리하세요:
    # return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
f(x)

# 유용한 패턴은 정적인 연산(즉, 컴파일 시간에 수행되어야 함)에는 numpy를 사용하고, 
# 추적되어야 하는 연산(즉, 실행 시간에 컴파일되고 실행되어야 함)에는 jax.numpy를 
# 사용하는 것입니다. 
@jit
def f(x):
    print(f"x = {x}")
    print(f"x.shape = {x.shape}")
    print(f"np.prod(x.shape) = {np.prod(x.shape)}")
    return x.reshape((np.prod(x.shape),))

x = jnp.ones((2, 3))
f(x)