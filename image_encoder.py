"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import math

import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02: split the feature axis D=768 into H=12 heads of 64 values.
        # Step 1: reshape [B,N,D] -> [B,N,H,D/H], i.e. cut the feature axis
        #         into 12 consecutive chunks (head 0 = features 0..63, ...).
        # Step 2: transpose the head axis forward: [B,N,H,D/H] -> [B,H,N,D/H].
        # Token order N is preserved; only the feature axis is reorganized.
        x = x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q: [B,H,N,64], k: [B,H,N,64]. Swap ONLY the last two axes of k
        # (k is 4D, unlike the 2D miniature) so heads/images stay grouped:
        # k.transpose(0,1,3,2) -> [B,H,64,N]. Then a batched matmul gives
        # every (head, query, key) similarity score in one operation.
        # Scale by sqrt(head_dim)=8 so softmax gradients stay well-behaved.
        # NOTE: use math.sqrt (a plain Python float), NOT np.sqrt. NumPy 2.x
        # promotes float32 arrays to float64 when dividing by a NumPy float64
        # scalar, which silently breaks every later stage's dtype check.
        # scores: [B,H,N,N], float32 throughout.
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # coefficients: [B,H,N,N] and v: [B,H,N,64] -> context: [B,H,N,64];
        # each query row becomes its weighted sum over all key rows.
        context = coefficients @ v
        # Join the 12 heads back: [B,H,N,64] -> [B,N,H,64] -> [B,N,D].
        merged = context.transpose(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.width)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        # out_weight is stored [out,in] = [D,D], so multiply with .T:
        # merged [B,N,in] @ [in,out] -> [B,N,D], then add the bias rowwise.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # The attention result is an ADDITIVE update to x; both keep [B,N,D].
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        # Step 1: expand to the 4x-wide MLP layer, [B,N,D] -> [B,N,4D].
        #         fc1_weight is stored [out,in] = [4D,D], hence .T.
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        # Step 2: project back to the model width, [B,N,4D] -> [B,N,D].
        #         fc2_weight is stored [out,in] = [D,4D], hence .T again.
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        # Step 3: second residual addition completes the block.
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        # Step 1: cut H and W into a 7x32 grid: [B,3,224,224] -> [B,3,7,32,7,32].
        B = images.shape[0]
        grid = images.reshape(B, 3, 7, 32, 7, 32)
        # Step 2: reorder axes to visit patches left-to-right then
        #         top-to-bottom, with C,H,W inside each patch:
        #         [B,3,ph,pw,gh,gw] -> [B,gh,gw,3,ph,pw].
        grid = grid.transpose(0, 2, 4, 1, 3, 5)
        # Step 3: flatten the grid into 49 patch rows and C,H,W into 3072
        #         features. reshape returns float32 unchanged.
        return grid.reshape(B, 49, 3 * 32 * 32)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        # patch_weight was reshaped from conv1.weight to [out,in]=[768,3072];
        # the convolution with 32x32 kernels equals a matmul with .T, and no
        # bias term exists in this checkpoint.
        x = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        # Step 1: put the CLS row in front of the 49 patch rows:
        #         [B,1,768] + [B,49,768] -> [B,50,768].
        x = np.concatenate((cls, x), axis=1)
        # Step 2: add the learned [50,768] position rows; the leading axis
        #         broadcasts over the batch. Result stays float32.
        return x + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        # Step 1: take the row-0 (CLS) token of every image: [B,50,768] -> [B,768].
        cls_row = tokens[:, 0, :]
        # Step 2: the learned post-Norm (unlike ln_pre, applied only at the end).
        normalized = self.post_norm(cls_row)
        # Step 3: project 768 -> 512. self.projection is [in,out]=[768,512],
        #         the ONLY linear weight not stored transposed.
        output = normalized @ self.projection
        # Step 4: scale each row to unit length (the given helper).
        return unit_vector(output)

        return unit_vector(output)
