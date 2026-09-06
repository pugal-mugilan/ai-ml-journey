import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d_model = d_k = 8


def sentence_to_embeddings(sentence):
    words = sentence.split()
    seq_len = len(words)
    embeddings = np.random.randn(seq_len, d_model)
    return words, embeddings


def qkv_projection(X, d_k):
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    attention_scores = Q @ K.T
    scaled_scores = attention_scores / np.sqrt(d_k)

    probabilities = softmax(scaled_scores)

    weights = probabilities @ V

    return weights, probabilities


def multihead_attention(X, words, num_heads=4):
    d_k = d_model // num_heads

    head_attn_weights = []
    final_outputs = []
    for head in range(num_heads):
        output, attn_weights = qkv_projection(X, d_k)
        head_attn_weights.append(attn_weights)
        final_outputs.append(output)
    combined_weights = np.concatenate(final_outputs, axis=1)

    W_O = np.random.randn(d_model, d_model)
    weights = combined_weights @ W_O

    return head_attn_weights, weights


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def plot_attention_heatmap(head_attn_weights, words, title="Self-Attention"):
    num_heads = len(head_attn_weights)
    fig, axes = plt.subplots(1, num_heads,  figsize=(4 * num_heads, max(4, len(words) * 0.8)))

    if num_heads == 1:
        axes = [axes]

    for i, (ax, weights) in enumerate(zip(axes, head_attn_weights)):
        sns.heatmap(
            weights,
            xticklabels=words,
            yticklabels=words,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=False,
            square=True,
        )
        ax.set_title(f"Head {i + 1}", fontsize=12)
        ax.set_xlabel("Keys (attended to)")
        ax.set_ylabel("Queries (doing the attending)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# --- Run on a sentence ---
sentence = "the cat sat on the mat because it was tired"
labels, embeddings = sentence_to_embeddings(sentence)
head_weights, output = multihead_attention(embeddings, labels, 4)

fig = plot_attention_heatmap(head_weights, labels, f'"{sentence}"')
fig.savefig("attention_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: attention_heatmap.png")

# IF YOU TRY TO FIND A PATTERN IN THE attention_heatmap, you won't because
# patterns are random and non-repeatable because weights are untrained.
# Meaningful specialization requires backpropagation across millions of examples.