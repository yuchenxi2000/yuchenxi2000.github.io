"""
生日问题，班上n个人，求存在两人生日相同概率
"""
import matplotlib.pyplot as plt

days = 365


def prod(m, n):
    # m x (m-1) x ... x (n+1)
    p = 1
    x = m
    while x > n:
        p *= x
        x -= 1
    return p


def prob(n):
    return 1.0 - prod(days, days - n) / pow(days, n)


persons = []
probs = []
for i in range(1, 106):
# for i in range(1, days + 1):
    persons.append(i)
    probs.append(prob(i))

plt.plot(persons, probs)
plt.axhline(1, c='r')
plt.xlabel('Number of students')
plt.ylabel('Probability')
plt.tight_layout()
plt.savefig('fig.png', dpi=300)
# plt.show()
