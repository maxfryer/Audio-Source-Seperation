import string
import random
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *


def generate_id() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=12))


@dataclass
class Person:
    name: str
    address: str
    active: bool = True
    email_addresses: list[str] = field(default_factory=list)
    id: str = field(init=False, default_factory=generate_id)


def main() -> None:
    person = Person(name="John", address="123 Main St")
    print(person)


x = np.linspace(0, 100, 100)
y = [np.random.randint(30) for i in range(100)]


def mplplot() -> None:
    plt.plot(x, y)
    plt.show()


def gg_plot():
    data = {'X': x, 'Y': y}
    df = pd.DataFrame(data)
    g = ggplot(df, aes(x=x, y=y)) + geom_line()
    return g
    #
    # fig = g.duraw()
    # fig.show()

if __name__ == '__main__':
    # main()
    plot = gg_plot()
    print(plot)

