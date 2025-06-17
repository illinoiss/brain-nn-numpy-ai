import json
import os
import re
import random
import numpy as np

BRAIN_FILE = "brain.json"
CONFIDENCE_THRESHOLD = 0.7

# ———————————————— Loading/Saving the Knowledge Base ————————————————
if os.path.exists(BRAIN_FILE):
    with open(BRAIN_FILE, "r", encoding="utf-8") as f:
        try:
            brain = json.load(f)
        except json.JSONDecodeError:
            brain = {"memory": {"name": None}, "knowledge": []}
else:
    brain = {"memory": {"name": None}, "knowledge": []}

def save_brain():
    try:
        with open(BRAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(brain, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving the knowledge base: {e}")

# ———————————————— Tokenization and Vocabulary ————————————————
def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def build_vocab(knowledge):
    vocab = set()
    for item in knowledge:
        inputs = item.get("input", [])
        outputs = item.get("output", [])
        if isinstance(inputs, list):
            for phrase in inputs:
                vocab.update(tokenize(phrase))
        elif isinstance(inputs, str):
            vocab.update(tokenize(inputs))
        if isinstance(outputs, list):
            for ans in outputs:
                vocab.update(tokenize(ans))
    return sorted(vocab)

def vectorize(text, vocab):
    tokens = tokenize(text)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab.index(t)] += 1
    return vec

# ———————————————— Neural Network Class ————————————————
class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.uniform(-1, 1, (hidden_size, input_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.uniform(-1, 1, (output_size, hidden_size)).astype(np.float32)
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, y):
        return y * (1 - y)

    def forward(self, x):
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y_true):
        y_pred = self.a2
        error = y_true - y_pred
        d_output = error * self.dsigmoid(y_pred)
        d_hidden = np.dot(self.W2.T, d_output) * self.dsigmoid(self.a1)

        self.W2 += self.lr * np.outer(d_output, self.a1)
        self.b2 += self.lr * d_output
        self.W1 += self.lr * np.outer(d_hidden, x)
        self.b1 += self.lr * d_hidden

        return np.mean(error ** 2)

# ———————————————— Preparing the Data ————————————————
def prepare_training_data(brain, vocab):
    X, Y, outputs = [], [], []
    for item in brain["knowledge"]:
        inputs = item.get("input", [])
        answers = item.get("output", [])
        questions = inputs if isinstance(inputs, list) else [inputs]
        for q in questions:
            x_vec = vectorize(q, vocab)
            for ans in answers:
                y_vec = vectorize(ans, vocab)
                X.append(x_vec)
                Y.append(y_vec)
                outputs.append(ans)
    if not X:
        return np.array([]), np.array([]), []
    return np.vstack(X), np.vstack(Y), outputs

# ———————————————— Training the Model ————————————————
def train_network(brain):
    vocab = build_vocab(brain["knowledge"])
    X_train, Y_train, outputs = prepare_training_data(brain, vocab)
    if X_train.size == 0 or Y_train.size == 0:
        return None, vocab, outputs, []
    net = NeuralNet(input_size=len(vocab), hidden_size=64, output_size=len(vocab), lr=0.01)
    outputs_vec = [vectorize(ans, vocab) for ans in outputs]
    for epoch in range(300):
        mse = 0
        for x, y in zip(X_train, Y_train):
            net.forward(x)
            mse += net.backward(x, y)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, MSE={mse/len(X_train):.5f}")
    return net, vocab, outputs, outputs_vec

# ———————————————— Prediction and Learning in Dialogue ————————————————
def predict_answer(net, x_vec, outputs, outputs_vec):
    out_vec = net.forward(x_vec)
    sims = []
    for ans_vec in outputs_vec:
        dot = np.dot(out_vec, ans_vec)
        norm = np.linalg.norm(out_vec) * np.linalg.norm(ans_vec)
        sims.append(dot / (norm + 1e-10))
    idx = np.argmax(sims)
    return outputs[idx] if sims[idx] >= CONFIDENCE_THRESHOLD else None

def add_knowledge(question, answers, brain):
    q = question.lower().strip()
    for item in brain["knowledge"]:
        inputs = item.get("input", [])
        if (isinstance(inputs, list) and q in inputs) or (isinstance(inputs, str) and inputs == q):
            for a in answers:
                if a not in item["output"]:
                    item["output"].append(a)
            return
    brain["knowledge"].append({"input": [q], "output": answers})

def learn_from_input(text, brain):
    if text.lower().startswith("teach:"):
        parts = text[6:].split("—")
        if len(parts) != 2:
            return "Invalid format. Use: teach: question — answer1, answer2"
        q = parts[0].strip()
        ans = [a.strip() for a in parts[1].split(",") if a.strip()]
        if not ans:
            return "Specify at least one answer."
        add_knowledge(q, ans, brain)
        save_brain()
        return f"Learned: '{q}' → {', '.join(ans)}"
    return None

# ———————————————— Chat Loop ————————————————
def chat():
    net, vocab, outputs, outputs_vec = train_network(brain)
    if net is None:
        print("AI: I don't know anything yet. Teach me, for example:\nteach: hello — hi, hey")
    else:
        print("AI: I'm ready to chat! Type 'bye' to exit.")

    while True:
        user = input("You: ")
        text = user.strip().lower()

        # Exit command
        if text in ["bye", "exit"]:
            print("AI: Goodbye!")
            break

        # 1) Priority lookup in knowledge base
        matched = False
        for item in brain["knowledge"]:
            inputs = item.get("input")
            if isinstance(inputs, str):
                inputs = [inputs]
            for phrase in inputs:
                if phrase in text:
                    print("AI:", random.choice(item["output"]))
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue

        # 2) Learning new knowledge
        learned = learn_from_input(user, brain)
        if learned:
            print("AI:", learned)
            net, vocab, outputs, outputs_vec = train_network(brain)
            continue

        # 3) Remember user’s name
        if "my name is" in text:
            name = text.split("my name is")[-1].strip().capitalize()
            brain["memory"]["name"] = name
            save_brain()
            print(f"AI: Nice to meet you, {name}!")
            continue

        if "what is my name" in text:
            nm = brain["memory"].get("name")
            if nm:
                print(f"AI: You told me your name is {nm}.")
            else:
                print("AI: I don't know your name yet. You can say 'my name is ...'")
            continue

        # 4) Neural network prediction
        if net:
            x_vec = vectorize(user, vocab)
            ans = predict_answer(net, x_vec, outputs, outputs_vec)
            if ans:
                print("AI:", ans)
                continue

        # 5) Fallback
        print("AI: I don't know how to answer that. You can teach me by typing:\nteach: question — answer1, answer2")
if __name__ == "__main__":
    try:
        chat()
    except Exception as e:
        print(f"An error occurred: {e}")