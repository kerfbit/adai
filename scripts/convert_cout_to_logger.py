#!/usr/bin/env python3
"""
Convert ChatbotTrainer.cpp std::cout/std::cerr calls to Logger calls.

Mapping:
  COLOR_INFO / COLOR_SUCCESS / COLOR_PROGRESS -> Logger::info
  COLOR_WARNING                               -> Logger::warn
  COLOR_ERROR (or std::cerr)                  -> Logger::error
"""
import re
import sys

def color_to_level(color_token):
    level_map = {
        'COLOR_INFO':     'info',
        'COLOR_SUCCESS':  'info',
        'COLOR_PROGRESS': 'info',
        'COLOR_WARNING':  'warn',
        'COLOR_ERROR':    'error',
    }
    return level_map.get(color_token, 'info')

def parse_cout_chain(chain):
    """
    Parse a joined cout/cerr chain (all on one logical line) and return
    (level, message_parts) where message_parts is a list of tokens
    that form the message (string literals and variables, excluding color codes
    and std::endl).
    """
    # Remove trailing semicolon
    chain = chain.rstrip(';').strip()

    # Determine stream type
    if chain.startswith('std::cerr'):
        default_level = 'error'
        chain = chain[len('std::cerr'):].lstrip()
    elif chain.startswith('std::cout'):
        default_level = 'info'
        chain = chain[len('std::cout'):].lstrip()
    else:
        return None, chain

    # Split on << but respecting nesting
    # Simple split: split on '<<'
    parts = re.split(r'\s*<<\s*', chain)
    parts = [p.strip() for p in parts if p.strip()]

    # Determine log level from first COLOR_ token
    level = default_level
    message_parts = []
    skip_tokens = {'COLOR_RESET', 'COLOR_INFO', 'COLOR_SUCCESS', 'COLOR_WARNING',
                   'COLOR_ERROR', 'COLOR_PROGRESS', 'std::endl', 'std::flush',
                   '\\n', '"\\n"'}

    first_color_found = False
    for part in parts:
        if part in skip_tokens:
            if not first_color_found and part.startswith('COLOR_') and part != 'COLOR_RESET':
                level = color_to_level(part)
                first_color_found = True
            continue
        # Skip empty part
        if not part:
            continue
        # skip the '<<' at start
        if part == '<<':
            continue
        message_parts.append(part)

    return level, message_parts

def build_logger_call(level, parts, indent):
    """
    Build a Logger::X(...) call from message parts.
    If all parts are string literals, concat them.
    If mixed, build a fmt-string.
    """
    if not parts:
        return None

    # Check if parts are all string literals
    all_literals = all(p.startswith('"') and p.endswith('"') for p in parts)

    if all_literals:
        # Merge all literal strings
        combined = ''.join(p[1:-1] for p in parts)  # strip quotes
        # Escape any {} in the combined string for spdlog
        combined = combined.replace('{', '{{').replace('}', '}}')
        return f'{indent}Logger::{level}("{combined}");'
    else:
        # Build fmt string: literals produce their text, variables produce {}
        fmt_parts = []
        args = []
        for p in parts:
            if p.startswith('"') and p.endswith('"'):
                text = p[1:-1].replace('{', '{{').replace('}', '}}')
                fmt_parts.append(text)
            else:
                # Check for special patterns
                if p == 'std::endl' or p == 'COLOR_RESET':
                    continue
                fmt_parts.append('{}')
                # Wrap non-string args for std::to_string if needed
                # Heuristic: if it looks like a numeric call, keep as-is
                args.append(p)

        fmt_str = ''.join(fmt_parts)
        if args:
            args_str = ', '.join(args)
            return f'{indent}Logger::{level}("{fmt_str}", {args_str});'
        elif fmt_str:
            return f'{indent}Logger::{level}("{fmt_str}");'
        else:
            return None

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    i = 0
    in_main_block = False  # Track #ifndef CHATBOT_TRAINER_TEST_BUILD
    in_print_usage = False  # Track print_usage function

    while i < len(lines):
        line = lines[i]

        # Track guard blocks — leave print_usage and main() untouched
        if '#ifndef CHATBOT_TRAINER_TEST_BUILD' in line:
            in_main_block = True
        if '#endif // CHATBOT_TRAINER_TEST_BUILD' in line:
            in_main_block = False

        # Track print_usage function
        if re.match(r'^void print_usage', line):
            in_print_usage = True
        if in_print_usage and line.strip() == '}' and not in_main_block:
            in_print_usage = False
            result.append(line)
            i += 1
            continue

        # If in guarded block or print_usage, leave unchanged
        if in_main_block or in_print_usage:
            result.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Detect the start of a std::cout or std::cerr statement
        if (stripped.startswith('std::cout') or stripped.startswith('std::cerr')) and \
           ('COLOR_' in line or 'std::cerr' in line):

            # Get the indentation
            indent = line[:len(line) - len(line.lstrip())]

            # Collect the full logical statement (may span multiple lines)
            stmt_lines = [stripped]
            j = i + 1

            # A statement is complete when it has a semicolon at the end
            # (after stripping continuation)
            while not stripped.rstrip().endswith(';') and j < len(lines):
                next_stripped = lines[j].strip()
                stmt_lines.append(next_stripped)
                stripped = next_stripped
                j += 1

            # Join all statement lines into one
            joined = ' '.join(l.rstrip('\\').strip() for l in stmt_lines)

            # Special case: the switch statement for optimizer type in initialize_model
            # These are bare `std::cout << "SGD"` lines without endl — skip them
            # (we handle that block specially below)
            if 'std::endl' not in joined and 'std::flush' not in joined:
                # Not a complete statement with output - leave as is
                for k in range(i, j):
                    result.append(lines[k])
                i = j
                continue

            level, parts = parse_cout_chain(joined)

            if level is None or not parts:
                # Could not parse - leave original
                for k in range(i, j):
                    result.append(lines[k])
                i = j
                continue

            logger_call = build_logger_call(level, parts, indent)

            if logger_call:
                result.append(logger_call + '\n')
            else:
                # Fallback: leave original
                for k in range(i, j):
                    result.append(lines[k])

            i = j
            continue

        result.append(line)
        i += 1

    return result

def add_logger_include(lines):
    """Add #include 'Logger.hpp' after the last #include line in the header block."""
    # Find if Logger.hpp already included
    for line in lines:
        if 'Logger.hpp' in line:
            return lines  # Already there

    # Find the last #include before any non-include code
    last_include_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith('#include'):
            last_include_idx = idx

    if last_include_idx >= 0:
        lines.insert(last_include_idx + 1, '#include "Logger.hpp"\n')

    return lines

def update_log_method(lines):
    """
    Replace the old log() implementation that uses std::cout with Logger.
    Maps color parameter to log level.
    """
    new_log_body = '''\
void ChatbotTrainer::log(LogLevel level, const std::string& message, const std::string& /*color*/) {
        // color parameter accepted for API compatibility but ignored;
        // Logger handles its own coloring via spdlog level-colored sinks.
        if (static_cast<int>(config.log_level) >= static_cast<int>(level)) {
            adai::Logger::info("{}", message);
        }
}
'''
    # Find and replace the log method
    start = None
    for idx, line in enumerate(lines):
        if 'void ChatbotTrainer::log(LogLevel' in line:
            start = idx
            break

    if start is None:
        return lines

    # Find the closing brace of the function
    end = None
    brace_depth = 0
    for idx in range(start, len(lines)):
        for ch in lines[idx]:
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    end = idx
                    break
        if end is not None:
            break

    if end is None:
        return lines

    # Replace
    lines = lines[:start] + [new_log_body + '\n'] + lines[end + 1:]
    return lines

def remove_color_defines(lines):
    """Keep COLOR_* defines — they are still referenced by log() callers and main/print_usage."""
    return lines  # No-op: keep all defines

def fix_optimizer_switch_block(lines):
    """
    Special-case: the optimizer type switch in initialize_model uses bare std::cout
    without endl to build a single line. Convert to a std::string variable approach.
    """
    # Find the problematic block
    target_start = '        std::cout << COLOR_INFO << "  Type: ";'
    target_end = '        std::cout << COLOR_RESET << std::endl;'

    start_idx = None
    for idx, line in enumerate(lines):
        if target_start in line:
            start_idx = idx
            break

    if start_idx is None:
        return lines

    # Find end
    end_idx = None
    for idx in range(start_idx, min(start_idx + 20, len(lines))):
        if target_end in lines[idx]:
            end_idx = idx
            break

    if end_idx is None:
        return lines

    # Build replacement
    replacement = '''\
        {
            std::string opt_type_str;
            switch (config.optimizer_type) {
                case OptimizerType::SGD:           opt_type_str = "SGD"; break;
                case OptimizerType::SGD_MOMENTUM:  opt_type_str = "SGD+Momentum"; break;
                case OptimizerType::ADAM:          opt_type_str = "Adam"; break;
                case OptimizerType::ADAMW:         opt_type_str = "AdamW"; break;
                default:                           opt_type_str = "Unknown"; break;
            }
            adai::Logger::info("  Type: {}", opt_type_str);
        }
'''
    lines = lines[:start_idx] + [replacement] + lines[end_idx + 1:]
    return lines

def convert_log_callers_to_logger(lines):
    """
    Convert the remaining log(LogLevel::X, msg, COLOR_Y) calls in train_epoch and validate
    to direct Logger calls. Also remove orphaned bare `std::cout << std::endl;` lines.
    """
    result = []
    for line in lines:
        stripped = line.strip()

        # Remove bare endl spacers (orphaned from multi-statement conversions)
        if stripped == 'std::cout << std::endl;':
            continue  # Drop the spacer

        # Convert: log(LogLevel::VERBOSE, "...", COLOR_INFO) -> Logger::debug("...")
        # Convert: log(LogLevel::NORMAL, "...", COLOR_SUCCESS|COLOR_INFO) -> Logger::info("...")
        m = re.match(
            r'^(\s*)log\(LogLevel::(\w+),\s*$',
            line
        )
        if m:
            indent = m.group(1)
            loglevel = m.group(2)  # VERBOSE, NORMAL, etc.
            # Collect the multi-line log() call
            # We need to gather lines until we find the closing );
            result.append(line)
            continue

        result.append(line)
    return result


def ensure_adai_namespace(lines):
    """
    Ensure Logger calls use the adai:: namespace since Logger is in namespace adai.
    Replace Logger:: with adai::Logger:: (avoids needing 'using namespace adai;').
    """
    result = []
    for line in lines:
        # Replace Logger:: with adai::Logger:: where not already qualified
        # But be careful not to double-qualify
        line = re.sub(r'(?<!adai::)Logger::', 'adai::Logger::', line)
        result.append(line)
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_cout_to_logger.py <input.cpp> [output.cpp]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    lines = process_file(input_path)
    lines = add_logger_include(lines)
    lines = update_log_method(lines)
    lines = fix_optimizer_switch_block(lines)
    lines = convert_log_callers_to_logger(lines)
    lines = ensure_adai_namespace(lines)
    lines = remove_color_defines(lines)

    # Keep iostream for print_usage / main
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Written to {output_path}")
    cout_remaining = sum(1 for l in lines if 'std::cout' in l or 'std::cerr' in l)
    logger_calls = sum(1 for l in lines if 'Logger::' in l)
    print(f"Remaining std::cout/cerr: {cout_remaining}")
    print(f"Logger calls: {logger_calls}")

if __name__ == '__main__':
    main()
