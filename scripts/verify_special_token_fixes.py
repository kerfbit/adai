#!/usr/bin/env python3

# @adai-status: experimental        (one-off verification script for a specific past fix)
# @adai-version: 0.3.0
# @adai-reviewed: 2026-09-07

"""Simple test to verify special token fixes work correctly"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and return the output"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("SPECIAL TOKEN FIXES VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Check tokenizer tests
    if not run_command(
        "cd /home/rodney/Repos/adai/build && make test ARGS='-R TokenizerTests -V'",
        "Tokenizer Unit Tests"
    ):
        all_passed = False
    
    # Test 2: Check that vocab file has correct special tokens
    print("\n" + "="*60)
    print("Testing: Vocabulary File Special Tokens")
    print("="*60)
    
    try:
        with open('/home/rodney/Repos/adai/vocab.txt', 'r') as f:
            content = f.read()
            
        checks = {
            'pad_token_id 0': 'pad_token_id 0' in content,
            'unk_token_id 1': 'unk_token_id 1' in content,
            'bos_token_id 2': 'bos_token_id 2' in content,
            'eos_token_id 3': 'eos_token_id 3' in content,
        }
        
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}")
            if not passed:
                all_passed = False
                
    except FileNotFoundError:
        print("✗ vocab.txt not found")
        all_passed = False
    
    # Test 3: Verify compile-time changes
    print("\n" + "="*60)
    print("Testing: Code Changes Present")
    print("="*60)
    
    code_checks = [
        ("/home/rodney/Repos/adai/src/BPETokenizer.hpp", "get_bos_token_id", "BPETokenizer getter methods"),
        ("/home/rodney/Repos/adai/src/EncoderDecoderModel.cpp", "gen_config.bos_token_id = 2", "Model default BOS=2"),
        ("/home/rodney/Repos/adai/src/EncoderDecoderModel.cpp", "gen_config.eos_token_id = 3", "Model default EOS=3"),
        ("/home/rodney/Repos/adai/src/EncoderDecoderModel.cpp", "tokenizer->get_bos_token_id()", "Token ID synchronization"),
        ("/home/rodney/Repos/adai/src/ChatbotTrainer.cpp", "encode(pair.input, false)", "Training encoder no special tokens"),
        ("/home/rodney/Repos/adai/src/ChatbotTrainer.cpp", "encode(pair.response, true)", "Training decoder with special tokens"),
    ]
    
    for filepath, search_string, description in code_checks:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            present = search_string in content
            status = "✓" if present else "✗"
            print(f"{status} {description}")
            if not present:
                all_passed = False
        except FileNotFoundError:
            print(f"✗ {filepath} not found")
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VERIFICATION CHECKS PASSED!")
        print("="*60)
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
