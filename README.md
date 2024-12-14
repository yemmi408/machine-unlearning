
# **Machine Unlearning: A Neural Network Learning Backwards** 🧠 

## What just happened here?!
This project is my humble contribution to the world of deep learning: a neural network that, for unknown reasons, **learns in reverse**. It starts with **great accuracy** and gradually forgets everything it once knew.

Essentially, I accidentally invented **Machine Unlearning**. A true milestone in the history of Artificial Intelligence (or lack thereof).

---

## 📸 **The Code in Action**
Here's how well my neural network "learns":
```
Epoch 1, Accuracy: 86.82%
Epoch 2, Accuracy: 87.07%
...
Epoch 20, Accuracy: 57.97%
```
Yes, you read that right. Each epoch makes things worse. The poor model started out strong and ended up begging for help. 🤖💔

---

## ⚙️ **How to Run This Project**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/machine-unlearning.git
   cd machine-unlearning
   ```

2. Compile the C code (if you're feeling brave):
   ```bash
   gcc -O3 -march=native -ffast-math -o nn nn.c -lm
   ```

3. Run it and witness the magic of unlearning:
   ```bash
   ./nn
   ```

4. Sit back and enjoy watching your model's performance decline.

---

## 📚 **How It Works?**

Honestly? Even I’m not quite sure. 

---

## **Why Does This Exist?**

Because making mistakes is human, but making a neural network unlearn is a **rare talent**.

---

## 🔗 **Based On**

This project is **based on** the amazing work at [miniMNIST-c](https://github.com/konrad-gajdus/miniMNIST-c) by [Konrad Gajdus](https://github.com/konrad-gajdus). I just accidentally broke it.

