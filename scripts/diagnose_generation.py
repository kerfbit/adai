#!/usr/bin/env python3
"""
Diagnostic script to analyze token generation issues
Helps identify why unknown tokens appear at beginning of responses
"""

import sys
import re

def analyze_vocab_file(vocab_path):
    """Analyze vocabulary file for potential issues."""
    print(f"Analyzing vocabulary file: {vocab_path}")
    print("="*70)
    
    special_tokens = {}
    vocab_tokens = {}
    issues = []
    
    with open(vocab_path, 'r', encoding='utf-8', errors='replace') as f:
        section = None
        line_num = 0
        
        for line in f:
            line_num += 1
            line = line.rstrip('\n')
            
            if not line or line.startswith('#'):
                continue
            
            # Section headers
            if line == "SPECIAL_TOKENS":
                section = "SPECIAL_TOKENS"
                continue
            elif line == "VOCAB":
                section = "VOCAB"
                continue
            elif line.startswith("BPE_MERGES"):
                section = "BPE_MERGES"
                break  # We don't need to analyze BPE merges
            
            # Parse special tokens
            if section == "SPECIAL_TOKENS":
                parts = line.split()
                if len(parts) == 2:
                    special_tokens[parts[0]] = int(parts[1])
            
            # Parse vocab
            elif section == "VOCAB":
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, token_id = parts[0], parts[1]
                        try:
                            vocab_tokens[token] = int(token_id)
                        except ValueError:
                            issues.append(f"Line {line_num}: Invalid token ID: {token_id}")
    
    # Check special tokens
    print("\n1. SPECIAL TOKENS")
    print("-" * 70)
    required = ['pad_token_id', 'unk_token_id', 'bos_token_id', 'eos_token_id']
    for req in required:
        if req in special_tokens:
            print(f"✓ {req}: {special_tokens[req]}")
        else:
            print(f"✗ {req}: MISSING")
            issues.append(f"Missing special token: {req}")
    
    # Check if special tokens are in vocab
    print("\n2. SPECIAL TOKEN PRESENCE IN VOCAB")
    print("-" * 70)
    special_token_names = ['<pad>', '<unk>', '<bos>', '<eos>']
    for token_name in special_token_names:
        if token_name in vocab_tokens:
            expected_id = {
                '<pad>': special_tokens.get('pad_token_id'),
                '<unk>': special_tokens.get('unk_token_id'),
                '<bos>': special_tokens.get('bos_token_id'),
                '<eos>': special_tokens.get('eos_token_id')
            }[token_name]
            
            actual_id = vocab_tokens[token_name]
            if actual_id == expected_id:
                print(f"✓ {token_name}: {actual_id} (matches special token ID)")
            else:
                print(f"✗ {token_name}: {actual_id} (expected {expected_id})")
                issues.append(f"Token {token_name} has ID {actual_id} but should be {expected_id}")
        else:
            print(f"✗ {token_name}: NOT IN VOCAB")
            issues.append(f"Special token {token_name} not in vocabulary")
    
    # Check for duplicate IDs
    print("\n3. DUPLICATE TOKEN IDS")
    print("-" * 70)
    id_counts = {}
    for token, token_id in vocab_tokens.items():
        id_counts[token_id] = id_counts.get(token_id, 0) + 1
    
    duplicates = {id: count for id, count in id_counts.items() if count > 1}
    if duplicates:
        print(f"✗ Found {len(duplicates)} duplicate IDs:")
        for token_id, count in sorted(duplicates.items())[:10]:
            tokens_with_id = [t for t, tid in vocab_tokens.items() if tid == token_id]
            print(f"  ID {token_id}: {count} tokens - {tokens_with_id[:3]}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
    else:
        print("✓ No duplicate token IDs found")
    
    # Summary
    print("\n4. SUMMARY")
    print("=" * 70)
    print(f"Total vocabulary size: {len(vocab_tokens)}")
    print(f"Special tokens defined: {len(special_tokens)}")
    
    if issues:
        print(f"\n⚠ ISSUES FOUND: {len(issues)}")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False
    else:
        print("\n✓ Vocabulary appears correctly formatted")
        return True


def suggest_fixes():
    """Suggest fixes for common generation issues."""
    print("\n" + "=" * 70)
    print("COMMON FIXES FOR <UNK> TOKEN GENERATION")
    print("=" * 70)
    
    print("""
1. RECOMPILE AFTER VOCAB FIX
   After fixing vocabulary, you must recompile the code:
   
   cd /home/rodney/Repos/adai
   ./build.sh
   
2. RETRAIN FROM SCRATCH
   Do NOT try to resume training with old checkpoints:
   
   ./build/bin/chatbot_trainer \\
       --data sample_training_data.txt \\
       --vocab vocab_fixed.txt \\
       --output new_model.bin \\
       --epochs 10
   
3. CHECK TRAINING DATA QUALITY
   Ensure training data has good coverage of vocabulary:
   
   # Check if training data uses diverse vocabulary
   cat sample_training_data.txt | wc -l
   
4. INCREASE TRAINING EPOCHS
   Model may need more training to learn vocabulary:
   
   --epochs 20  # Instead of 10
   
5. CHECK FOR VOCAB-MODEL MISMATCH
   Verify you're using matching vocab and model files:
   
   ls -lh vocab.txt chatbot_model.bin*
   
6. TEST WITH SIMPLE INPUT
   Try very simple inputs to see if model generates ANY valid tokens:
   
   ./build/bin/chatbot vocab.txt chatbot_model.bin
   Input: "Hello"
   
7. DEBUG GENERATION
   Add verbose logging to see what tokens are being generated:
   
   export DEBUG_GENERATION=1
   ./build/bin/chatbot vocab.txt chatbot_model.bin
""")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_generation.py vocab.txt")
        sys.exit(1)
    
    vocab_file = sys.argv[1]
    
    try:
        is_valid = analyze_vocab_file(vocab_file)
        suggest_fixes()
        
        if not is_valid:
            print("\n⚠ Please fix vocabulary issues before training")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {vocab_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
