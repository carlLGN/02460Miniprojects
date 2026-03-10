import numpy as np

def curve_length(curve: np.ndarray, resolution: int = 100_000) -> np.float16:
    """Implements 4.2 in LMLG.
    Takes curve on the form of an ndarray of lambda functions.

    for example
    c = np.array([
                lambda x: 2*x,
                lambda x: x + 10
                ])
    """


    vector_call = np.vectorize(lambda f, t: f(t))

    points = np.linspace(0, 1, resolution+1)

    c = vector_call(curve[:, None], points)
    c = c.T

    segment_lengths = np.linalg.norm(c[:-1]-c[1:], axis=1)
    length = np.sum(segment_lengths)


    return length


if __name__ == "__main__":
    c = np.array([
                lambda t: 2*t + 1,
                lambda t: -t**2
                ])
    
    print(curve_length(c))