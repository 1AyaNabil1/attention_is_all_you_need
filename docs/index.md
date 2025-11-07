---
layout: home
title: Home
---



<div class="section">
  <h2>About the Paper</h2>
  <p>
    "Attention Is All You Need" (Vaswani et al., 2017) introduced the Transformer architecture, 
    revolutionizing natural language processing and becoming the foundation for modern AI systems 
    like GPT, BERT, and beyond. The paper demonstrated that attention mechanisms alone, without 
    recurrence or convolution, could achieve state-of-the-art results in machine translation.
  </p>
  <p>
    This project implements the complete Transformer architecture from scratch in PyTorch, 
    faithfully following the paper's specifications for educational and research purposes.
  </p>
</div>

<div class="section">
  <h2>Implementation Highlights</h2>
  <div class="features">
    <div class="feature-card">
      <h3>Complete Architecture</h3>
      <p>Full encoder-decoder implementation with multi-head attention, positional encoding, and feed-forward networks.</p>
    </div>
    
    <div class="feature-card">
      <h3>Paper Faithful</h3>
      <p>Follows the original paper's specifications with default hyperparameters (d_model=512, N=6, h=8).</p>
    </div>
    
    <div class="feature-card">
      <h3>Training Pipeline</h3>
      <p>Complete training infrastructure with validation, checkpointing, and TensorBoard visualization.</p>
    </div>
    
    <div class="feature-card">
      <h3>Bilingual Translation</h3>
      <p>Trained on English-French dataset using OPUS Books corpus for machine translation tasks.</p>
    </div>
    
    <div class="feature-card">
      <h3>Pure PyTorch</h3>
      <p>Clean, readable implementation using only PyTorch without external transformer libraries.</p>
    </div>
    
    <div class="feature-card">
      <h3>Evaluation Metrics</h3>
      <p>Includes BLEU, CER, and WER metrics for comprehensive translation quality assessment.</p>
    </div>
  </div>
</div>

<div class="section">
  <h2>Architecture Overview</h2>
  <p>
    The Transformer architecture consists of an encoder-decoder structure with the following key components:
  </p>
  <div class="architecture">
    <table>
      <thead>
        <tr>
          <th>Component</th>
          <th>Value</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Model Dimension</td>
          <td>512</td>
          <td>Embedding and hidden state dimension</td>
        </tr>
        <tr>
          <td>Encoder/Decoder Layers</td>
          <td>6 each</td>
          <td>Stacked layers for processing</td>
        </tr>
        <tr>
          <td>Attention Heads</td>
          <td>8</td>
          <td>Multi-head attention mechanism</td>
        </tr>
        <tr>
          <td>Feed-Forward Dimension</td>
          <td>2048</td>
          <td>Inner layer dimension in FFN</td>
        </tr>
        <tr>
          <td>Dropout Rate</td>
          <td>0.1</td>
          <td>Regularization parameter</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="section">
  <h2>Key Features</h2>
  <ul>
    <li><strong>Multi-Head Self-Attention:</strong> Allows the model to attend to different representation subspaces simultaneously</li>
    <li><strong>Positional Encoding:</strong> Injects sequence order information using sine and cosine functions</li>
    <li><strong>Masked Attention:</strong> Prevents the decoder from attending to future positions during training</li>
    <li><strong>Layer Normalization:</strong> Applied before each sub-layer for training stability</li>
    <li><strong>Residual Connections:</strong> Facilitates gradient flow through deep networks</li>
  </ul>
</div>

<div class="section">
  <h2>Resources</h2>
  <p>
    <strong>Original Paper:</strong> <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a> (Vaswani et al., NeurIPS 2017)
  </p>
  <p>
    <strong>Source Code:</strong> <a href="https://github.com/1AyaNabil1/attention_is_all_you_need" target="_blank">GitHub Repository</a> - Complete implementation with training scripts and documentation
  </p>
</div>

<div class="section">
  <h2>Citation</h2>
  <div class="code-block">
    <pre><code>@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
          Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}</code></pre>
  </div>
</div>
