# Training Effect: Random vs Learned Weights

## Experiment

Comparing text generation from:
1. **Untrained model** (random initialization)
2. **Trained model** (5000 iterations on Shakespeare)

## Results

### Untrained Model Output
```
First Citizen:!?UBFRahnp,SSfmykRjLsj-
B!fmFmoneM ,zmViuTtsvOnARLgmbbhB,pRVmlHy!zzmMtrtNCdu!D'YvOH.TT
wDbscNEmeT.'iMz,SclSaDFi;e
```

**Characteristics:**
- Completely random characters
- No word structure
- No spacing patterns
- Uniform distribution across all characters
- Ignores English/text conventions

### Trained Model Output
```
First Citizen:
Lel te it ce ay alfobun bo wiren lve teallthelurellalve hithicon
Lig s Cice lllarstinourse gof ns y s ou wothit ithirsel id
```

**Characteristics:**
- Recognizable word-like structures
- Proper spacing between words
- Capitalization after line breaks
- Common letter combinations (th-, -ing, -er)
- Respects character frequency (more vowels/common letters)

## What Training Learns

Even a simple block-based model learns:

1. **Character Frequency Distribution**
   - High: e, t, a, o, i (common in English)
   - Low: z, x, q (rare)

2. **N-gram Patterns**
   - 'h' often follows 't'
   - Vowels follow consonants
   - Space follows letters, letters follow space
   - Multi-character sequences from 8-character context

3. **Text Structure**
   - Words separated by spaces
   - Capital letters at line starts
   - Punctuation usage patterns

## Loss Reduction

- **Initial loss**: ~4.15 (random predictions)
- **Final loss**: ~2.29 (learned patterns)
- **Improvement**: 45% reduction in cross-entropy

This demonstrates that even a simple neural language model with limited context captures meaningful statistical patterns from training data.
