#!/usr/bin/env python3
"""
Test and Repair Script for vocab.txt
Validates BPE tokenizer vocabulary file structure and repairs issues.
"""

import sys
from collections import Counter
from typing import List, Tuple, Set, Dict

class VocabValidator:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.errors = []
        self.warnings = []
        self.repairs_made = []
        
        # Expected structure
        self.expected_header = "# BPE Tokenizer Vocabulary v1.0"
        self.special_token_names = ['pad_token_id', 'unk_token_id', 'bos_token_id', 'eos_token_id']
        
        # Data storage
        self.vocab_size = None
        self.special_tokens = {}
        self.vocab_entries = []  # List of (token, id) tuples
        
    def read_file(self) -> List[str]:
        """Read the vocab file with fallback encoding options."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(self.filepath, 'r', encoding=encoding, errors='replace') as f:
                    lines = f.readlines()
                if encoding != 'utf-8':
                    self.warnings.append(f"File read with {encoding} encoding (not UTF-8)")
                return lines
            except FileNotFoundError:
                self.errors.append(f"File not found: {self.filepath}")
                return []
            except Exception as e:
                if encoding == encodings[-1]:  # Last encoding attempt
                    self.errors.append(f"Error reading file with all attempted encodings: {e}")
                    return []
                continue  # Try next encoding
        
        return []
    
    def parse_file(self, lines: List[str]) -> bool:
        """Parse the vocab file and extract components."""
        if not lines:
            return False
        
        # Check header
        if lines[0].strip() != self.expected_header:
            self.errors.append(f"Invalid header. Expected: '{self.expected_header}', Got: '{lines[0].strip()}'")
        
        # Find VOCAB_SIZE
        vocab_size_found = False
        special_tokens_section = False
        vocab_section = False
        line_num = 1
        
        for i, line in enumerate(lines[1:], start=2):
            line = line.rstrip('\n')
            
            # Parse VOCAB_SIZE
            if line.startswith('VOCAB_SIZE'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        self.vocab_size = int(parts[1])
                        vocab_size_found = True
                    except ValueError:
                        self.errors.append(f"Line {i}: Invalid VOCAB_SIZE value: {parts[1]}")
                else:
                    self.errors.append(f"Line {i}: Invalid VOCAB_SIZE format")
                continue
            
            # Parse SPECIAL_TOKENS section
            if line == 'SPECIAL_TOKENS':
                special_tokens_section = True
                continue
            
            # Parse VOCAB section marker
            if line == 'VOCAB':
                special_tokens_section = False
                vocab_section = True
                continue
            
            # Parse special tokens
            if special_tokens_section:
                parts = line.split()
                if len(parts) == 2:
                    token_name, token_id = parts[0], parts[1]
                    try:
                        self.special_tokens[token_name] = int(token_id)
                    except ValueError:
                        self.errors.append(f"Line {i}: Invalid special token ID: {token_id}")
                elif line.strip():  # Non-empty line
                    self.errors.append(f"Line {i}: Invalid special token format: {line}")
            
            # Parse vocab entries
            if vocab_section and line.strip():
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, token_id = parts[0], parts[1]
                        try:
                            self.vocab_entries.append((token, int(token_id)))
                        except ValueError:
                            self.errors.append(f"Line {i}: Invalid vocab ID: {token_id}")
                    else:
                        self.errors.append(f"Line {i}: Invalid vocab entry format (wrong number of tabs)")
                else:
                    self.errors.append(f"Line {i}: Missing tab separator in vocab entry: {repr(line)}")
        
        if not vocab_size_found:
            self.errors.append("VOCAB_SIZE not found")
        
        return True
    
    def validate(self) -> bool:
        """Run all validation checks."""
        print("Running validation checks...\n")
        
        # Check special tokens
        missing_tokens = []
        for token_name in self.special_token_names:
            if token_name not in self.special_tokens:
                missing_tokens.append(token_name)
        
        if missing_tokens:
            self.errors.append(f"Missing special tokens: {', '.join(missing_tokens)}")
        
        # Check for duplicate token IDs
        if self.vocab_entries:
            token_ids = [entry[1] for entry in self.vocab_entries]
            id_counts = Counter(token_ids)
            duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
            if duplicates:
                self.errors.append(f"Duplicate token IDs found: {duplicates}")
        
        # Check for duplicate tokens
        tokens = [entry[0] for entry in self.vocab_entries]
        token_counts = Counter(tokens)
        duplicate_tokens = {token: count for token, count in token_counts.items() if count > 1}
        if duplicate_tokens:
            self.errors.append(f"Duplicate tokens found ({len(duplicate_tokens)} unique): {list(duplicate_tokens.keys())[:10]}...")
        
        # Check vocab size matches
        actual_vocab_count = len(self.vocab_entries)
        if self.vocab_size and actual_vocab_count != self.vocab_size:
            self.warnings.append(f"VOCAB_SIZE mismatch: declared={self.vocab_size}, actual={actual_vocab_count}")
        
        # Check ID range validity
        special_token_ids = set(self.special_tokens.values())
        all_ids = set(token_ids) | special_token_ids
        if all_ids:
            min_id, max_id = min(all_ids), max(all_ids)
            if self.vocab_size and max_id >= self.vocab_size:
                self.errors.append(f"Token ID {max_id} exceeds VOCAB_SIZE {self.vocab_size}")
        
        # Check for negative IDs
        negative_ids = [tid for tid in token_ids if tid < 0]
        if negative_ids:
            self.errors.append(f"Negative token IDs found: {negative_ids[:10]}")
        
        return len(self.errors) == 0
    
    def repair(self) -> bool:
        """Attempt to repair common issues."""
        print("\nAttempting repairs...\n")
        
        repairs_possible = True
        
        # Repair VOCAB_SIZE mismatch
        actual_vocab_count = len(self.vocab_entries)
        if self.vocab_size and actual_vocab_count != self.vocab_size:
            self.vocab_size = actual_vocab_count
            self.repairs_made.append(f"Updated VOCAB_SIZE to {actual_vocab_count}")
        
        # Remove duplicate token IDs (keep first occurrence)
        seen_ids = set()
        seen_tokens = set()
        deduped_entries = []
        
        for token, token_id in self.vocab_entries:
            if token_id not in seen_ids and token not in seen_tokens:
                deduped_entries.append((token, token_id))
                seen_ids.add(token_id)
                seen_tokens.add(token)
            else:
                if token_id in seen_ids:
                    self.repairs_made.append(f"Removed duplicate ID {token_id} for token '{token}'")
                if token in seen_tokens:
                    self.repairs_made.append(f"Removed duplicate token '{token}'")
        
        removed_count = len(self.vocab_entries) - len(deduped_entries)
        if removed_count > 0:
            self.vocab_entries = deduped_entries
            self.vocab_size = len(self.vocab_entries)
            self.repairs_made.append(f"Total duplicates removed: {removed_count}")
        
        return len(self.repairs_made) > 0
    
    def write_repaired_file(self, output_path: str = None):
        """Write repaired vocab file."""
        if output_path is None:
            output_path = self.filepath
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"{self.expected_header}\n")
                f.write(f"VOCAB_SIZE {self.vocab_size}\n")
                
                # Write special tokens
                f.write("SPECIAL_TOKENS\n")
                for token_name in self.special_token_names:
                    if token_name in self.special_tokens:
                        f.write(f"{token_name} {self.special_tokens[token_name]}\n")
                
                # Write vocab entries
                f.write("VOCAB\n")
                for token, token_id in self.vocab_entries:
                    f.write(f"{token}\t{token_id}\n")
            
            print(f"\n✓ Repaired file written to: {output_path}")
            return True
        except Exception as e:
            print(f"\n✗ Error writing repaired file: {e}")
            return False
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        print(f"\nFile: {self.filepath}")
        print(f"Declared VOCAB_SIZE: {self.vocab_size}")
        print(f"Actual vocab entries: {len(self.vocab_entries)}")
        print(f"Special tokens: {len(self.special_tokens)}")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n✗ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✓ No errors found!")
        
        if self.repairs_made:
            print(f"\n🔧 REPAIRS MADE ({len(self.repairs_made)}):")
            for repair in self.repairs_made:
                print(f"  - {repair}")
        
        print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and repair vocab.txt file')
    parser.add_argument('file', nargs='?', default='vocab.txt', help='Path to vocab.txt file')
    parser.add_argument('--repair', action='store_true', help='Attempt to repair issues')
    parser.add_argument('--output', '-o', help='Output path for repaired file (default: overwrite original)')
    parser.add_argument('--test-only', action='store_true', help='Only test, do not repair')
    
    args = parser.parse_args()
    
    print(f"BPE Vocabulary Validator and Repair Tool")
    print(f"{'='*70}\n")
    
    validator = VocabValidator(args.file)
    
    # Read and parse file
    lines = validator.read_file()
    if not lines:
        print("Failed to read file. Exiting.")
        sys.exit(1)
    
    validator.parse_file(lines)
    
    # Validate
    is_valid = validator.validate()
    
    # Repair if requested
    if args.repair and not args.test_only:
        if not is_valid or validator.warnings:
            validator.repair()
            output_path = args.output if args.output else args.file
            validator.write_repaired_file(output_path)
        else:
            print("\nNo repairs needed - file is valid!")
    
    # Print report
    validator.print_report()
    
    # Exit with appropriate code
    if is_valid and not validator.warnings:
        print("\n✓ Vocabulary file is valid!")
        sys.exit(0)
    elif not is_valid:
        print("\n✗ Vocabulary file has errors!")
        sys.exit(1)
    else:
        print("\n⚠ Vocabulary file has warnings but no critical errors.")
        sys.exit(0)


if __name__ == '__main__':
    main()
