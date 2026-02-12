#include "IncrementalTrainer.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <command> [options]\n\n";
        std::cout << "Commands:\n";
        std::cout << "  init <vocab> <model>         Initialize incremental trainer\n";
        std::cout << "  add <data_file>              Add new training data\n";
        std::cout << "  gutenberg <book_id> [pairs]  Download & add Gutenberg book (default: 500 pairs)\n";
        std::cout << "  gutenberg-batch <id1,id2...> Download multiple books\n";
        std::cout << "  train <epochs>               Train on pending data\n";
        std::cout << "  retrain <epochs>             Full retrain on all data\n";
        std::cout << "  resume                       Resume from last session\n";
        std::cout << "  status                       Show training status\n";
        std::cout << "  history                      Show session history\n";
        std::cout << "\nPopular Gutenberg Books:\n";
        std::cout << "  1342  - Pride and Prejudice (Jane Austen)\n";
        std::cout << "  11    - Alice in Wonderland (Lewis Carroll)\n";
        std::cout << "  84    - Frankenstein (Mary Shelley)\n";
        std::cout << "  1661  - Sherlock Holmes (Arthur Conan Doyle)\n";
        std::cout << "  2701  - Moby Dick (Herman Melville)\n";
        std::cout << "  16328 - Beowulf\n";
        std::cout << "  1260  - Jane Eyre (Charlotte Bronte)\n";
        std::cout << "  98    - A Tale of Two Cities (Charles Dickens)\n";
        std::cout << "\nExample workflow:\n";
        std::cout << "  # Initial training\n";
        std::cout << "  " << argv[0] << " init vocab.txt chatbot_model.bin\n";
        std::cout << "  " << argv[0] << " gutenberg 1342 500\n";
        std::cout << "  " << argv[0] << " train 10\n";
        std::cout << "\n  # Add multiple classic books\n";
        std::cout << "  " << argv[0] << " gutenberg-batch 11,84,1661,2701\n";
        std::cout << "  " << argv[0] << " train 5\n";
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "init") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " init <vocab> <model>\n";
            return 1;
        }
        
        std::string vocab_path = argv[2];
        std::string model_path = argv[3];
        
        IncrementalConfig config;
        config.base_config.num_epochs = 10;
        config.base_config.learning_rate = 0.0001f;
        config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
        config.auto_save_enabled = true;
        config.auto_save_every_minutes = 30;
        config.auto_save_every_samples = 1000;
        
        IncrementalTrainer trainer(vocab_path, model_path, config);
        
        std::cout << "✅ Incremental trainer initialized\n";
        std::cout << "📁 Session directory: " << config.session_dir << "\n";
        
    } else if (command == "add") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " add <data_file>\n";
            return 1;
        }
        
        std::string data_file = argv[2];
        
        // Load trainer state
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        if (trainer.add_new_data(data_file)) {
            std::cout << "✅ Data file added to pending queue\n";
            std::cout << "📊 Pending files: " << trainer.get_pending_data_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add data file\n";
            return 1;
        }
        
    } else if (command == "gutenberg") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " gutenberg <book_id> [num_pairs]\n";
            std::cerr << "Example: " << argv[0] << " gutenberg 1342 500\n";
            return 1;
        }
        
        int book_id = std::stoi(argv[2]);
        int num_pairs = (argc >= 4) ? std::stoi(argv[3]) : 500;
        
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        std::cout << "📚 Downloading Project Gutenberg book #" << book_id << "...\n";
        
        if (trainer.add_gutenberg_book(book_id, num_pairs)) {
            std::cout << "✅ Book added to training queue (" << num_pairs << " pairs)\n";
            std::cout << "📊 Pending files: " << trainer.get_pending_data_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add Gutenberg book\n";
            return 1;
        }
        
    } else if (command == "gutenberg-batch") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " gutenberg-batch <id1,id2,id3,...> [num_pairs_each]\n";
            std::cerr << "Example: " << argv[0] << " gutenberg-batch 1342,11,84,1661 300\n";
            return 1;
        }
        
        std::string ids_str = argv[2];
        int num_pairs_each = (argc >= 4) ? std::stoi(argv[3]) : 500;
        
        // Parse comma-separated IDs
        std::vector<int> book_ids;
        std::stringstream ss(ids_str);
        std::string id;
        while (std::getline(ss, id, ',')) {
            book_ids.push_back(std::stoi(id));
        }
        
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        std::cout << "📚 Downloading " << book_ids.size() << " Project Gutenberg books...\n";
        
        if (trainer.add_gutenberg_books(book_ids, num_pairs_each)) {
            std::cout << "✅ Books added to training queue\n";
            std::cout << "📊 Pending files: " << trainer.get_pending_data_files().size() << "\n";
        } else {
            std::cerr << "❌ Failed to add Gutenberg books\n";
            return 1;
        }
        
    } else if (command == "train") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " train <epochs>\n";
            return 1;
        }
        
        int epochs = std::stoi(argv[2]);
        
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        if (trainer.get_pending_data_files().empty()) {
            std::cout << "⚠️  No pending data. Use 'add' command first.\n";
            return 1;
        }
        
        std::cout << "🚀 Starting incremental training for " << epochs << " epochs...\n";
        
        if (trainer.train_incremental(epochs)) {
            std::cout << "✅ Training completed successfully!\n";
            trainer.print_training_summary();
        } else {
            std::cerr << "❌ Training failed\n";
            return 1;
        }
        
    } else if (command == "retrain") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " retrain <epochs>\n";
            return 1;
        }
        
        int epochs = std::stoi(argv[2]);
        
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        std::cout << "🔄 Starting full retrain for " << epochs << " epochs...\n";
        
        if (trainer.train_full_retrain(epochs)) {
            std::cout << "✅ Full retrain completed!\n";
            trainer.print_training_summary();
        } else {
            std::cerr << "❌ Full retrain failed\n";
            return 1;
        }
        
    } else if (command == "resume") {
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        if (trainer.resume_last_session()) {
            std::cout << "✅ Resumed from last session\n";
            std::cout << "📂 Latest checkpoint: " << trainer.get_latest_checkpoint() << "\n";
        } else {
            std::cerr << "❌ Failed to resume session\n";
            return 1;
        }
        
    } else if (command == "status") {
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        trainer.print_training_summary();
        
        std::cout << "\n📋 Pending data files:\n";
        for (const auto& file : trainer.get_pending_data_files()) {
            std::cout << "  - " << file << "\n";
        }
        
    } else if (command == "history") {
        IncrementalTrainer trainer("vocab.txt", "chatbot_model.bin");
        
        trainer.print_session_history();
        trainer.print_data_registry();
        
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        std::cerr << "Run '" << argv[0] << "' without arguments for help\n";
        return 1;
    }
    
    return 0;
}
