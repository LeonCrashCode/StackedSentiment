#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/program_options.hpp>

#include <unordered_map>
#include <unordered_set>

using namespace std;
using namespace dynet;
namespace po = boost::program_options;

//float pdrop = 0.02;
float pdrop = 0.5;
float unk_prob = 0.1;
bool DEBUG = 0;

unsigned WORD_DIM = 200;
unsigned HIDDEN_DIM = 150;
unsigned TAG_HIDDEN_DIM = 64;
unsigned LAYERS = 1;
unsigned VOCAB_SIZE = 0;


float noscore = 10000;
dynet::Dict wd;

int kUNK; //tzy
unordered_map<unsigned, vector<float> > pretrained;
vector<float> unk_embedding;

unordered_map<unsigned, float> lex_dict;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data", po::value<string>(), "Test corpus")
        ("pdrop", po::value<float>()->default_value(0.5), "dropout probabilty")
	("unk_prob,u", po::value<float>()->default_value(0.1), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("word_dim", po::value<unsigned>()->default_value(200), "word embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(150), "hidden dimension")
        ("tag_hidden_dim", po::value<unsigned>()->default_value(64), "tag hidden dimension")
	("layers", po::value<unsigned>()->default_value(1), "layers")
	("train,t", "Should training be run?")
        ("pretrained,w", po::value<string>(), "Pretrained word embeddings")
        ("lexicon", po::value<string>(), "Sentiment Lexicon")
	("train_methods", po::value<unsigned>()->default_value(0), "0 for simple, 1 for mon, 2 for adagrad, 3 for adam")
	("report_i", po::value<unsigned>()->default_value(100), "report i")
        ("dev_report_i", po::value<unsigned>()->default_value(10), "dev report i")
	("count_limit", po::value<unsigned>()->default_value(50), "count limit")
	("debug", "Debug to output trace")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0 || conf->count("dev_data") == 0 || conf->count("test_data") == 0) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
}

void normalize_digital_lower(string& line){
  for(unsigned i = 0; i < line.size(); i ++){
    if(line[i] >= 'A' && line[i] <= 'Z'){
      line[i] = line[i] - 'A' + 'a';
    }
  }
}

class Instance{
public:
	vector<unsigned> raws;
	vector<unsigned> lows;

	vector<unsigned> words;
	vector<float> lex_score;
	unsigned label;
	
	Instance(){clear();};
        ~Instance(){};
	void clear(){
		raws.clear();
		lows.clear();
		words.clear();
		lex_score.clear();
	}	
	friend ostream& operator << (ostream& out, Instance& instance){
		for(unsigned i = 0; i < instance.raws.size(); i ++){
			out << wd.convert(instance.raws[i]) << "/"
			    << wd.convert(instance.lows[i]) << "/";
			if(instance.lex_score[i] == noscore) out<<"n ";
			else out<<instance.lex_score[i]<<" ";
		}
		out<<" ||| ";
		out<<instance.label;
		out<<"\n";
		return out;
	}
	void load(const string& line){
                istringstream in(line);
                string word;
                while(in>>word) {
                        if(word == "|||") break;
			raws.push_back(wd.convert(word));
			normalize_digital_lower(word);
			lows.push_back(wd.convert(word));		
                }
		in>>label;
		words = raws;
		lex_score.resize(words.size());
        }
	unsigned size(){assert(raws.size() == lows.size()); return raws.size();}
};

struct LSTMClassifier {
    LookupParameter p_word;

    Parameter p_l2rR;
    Parameter p_r2lR;
    Parameter p_bias;

    Parameter p_lbias;
    Parameter p_tag2label;

    Parameter p_start;
    Parameter p_end;

    LSTMBuilder l2rbuilder;
    LSTMBuilder r2lbuilder;

    float zero = 0;
    float one = 1.0;

    explicit LSTMClassifier(Model& model) :
        l2rbuilder(LAYERS, WORD_DIM , HIDDEN_DIM, &model),
        r2lbuilder(LAYERS, WORD_DIM , HIDDEN_DIM, &model)
    {
        p_word   = model.add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});

        p_l2rR = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
        p_r2lR = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
	p_bias = model.add_parameters({TAG_HIDDEN_DIM});

	p_tag2label = model.add_parameters({2, TAG_HIDDEN_DIM});
	p_lbias = model.add_parameters({2});

        p_start = model.add_parameters({WORD_DIM});
	p_end = model.add_parameters({WORD_DIM});

        for(auto& it : pretrained){
	    p_word.initialize(it.first, it.second);
        }
    }

    // return Expression of total loss
    Expression BuildGraph(Instance& inst, ComputationGraph& cg, float& num_correct, bool train) {
        const vector<unsigned>& sent = inst.words;
	unsigned label = inst.label;
        const unsigned slen = sent.size() ;

        l2rbuilder.new_graph(cg);  // reset builder for new graph
        l2rbuilder.start_new_sequence();

        r2lbuilder.new_graph(cg);  // reset builder for new graph
        r2lbuilder.start_new_sequence();

	Expression i_l2rR = parameter(cg, p_l2rR);
	Expression i_r2lR = parameter(cg, p_r2lR);
	Expression i_bias = parameter(cg, p_bias);
        
	Expression i_tag2label = parameter(cg, p_tag2label);
	Expression i_lbias = parameter(cg, p_lbias);

	Expression word_start = parameter(cg, p_start);
        Expression word_end = parameter(cg, p_end);

if(DEBUG)	cerr<<"sent size " << slen<<"\n";
        vector<Expression> i_words(slen);
        for (unsigned t = 0; t < slen; ++t) {
            i_words[t] = lookup(cg, p_word, sent[t]);
            if (train) i_words[t] = dropout(i_words[t], pdrop);
        }

if(DEBUG)	cerr<<"all input expression done\n";

        l2rbuilder.add_input(word_start);
        r2lbuilder.add_input(word_end);
        for (unsigned t = 0; t < slen; ++t) {
	    l2rbuilder.add_input(i_words[t]);
            r2lbuilder.add_input(i_words[slen - 1 - t]);
        }
	l2rbuilder.add_input(word_end);
        r2lbuilder.add_input(word_start);

if(DEBUG)	cerr<<"bilstm done\n";

        Expression i_r =  tanh(i_bias + i_l2rR * l2rbuilder.back() + i_r2lR * r2lbuilder.back());
	Expression i_r_t = tanh(i_lbias + i_tag2label * i_r);
	Expression output_loss = pickneglogsoftmax(i_r_t, label);

        auto prob_value = as_vector(cg.incremental_forward(i_r_t));
        float best = prob_value[0];
        unsigned bestk = 0;
        for(unsigned i = 1; i < prob_value.size(); i ++){
                if(best < prob_value[i]){best = prob_value[i]; bestk = i;}
        }
        if(bestk == label) num_correct += 1;

	return output_loss;
    }
};

void evaluate(vector<Instance>& instances, LSTMClassifier& lstmClassifier, float& acc)
{
    float num_correct = 0;
    float loss = 0;
    for (auto& sent : instances) {
        ComputationGraph cg;
        Expression nll = lstmClassifier.BuildGraph(sent, cg, num_correct, false);
        loss += as_scalar(cg.incremental_forward(nll));
    }
    acc = num_correct/ instances.size();
    cerr<<"Loss:"<< loss/ instances.size() << " ";
    cerr<<"Accuracy:"<< num_correct <<"/" << instances.size() <<" "<<acc<<" ";
}

int main(int argc, char** argv) {
    DynetParams dynet_params = extract_dynet_params(argc, argv);
    dynet_params.random_seed = 1989121013;
    dynet::initialize(dynet_params);
  
    cerr << "COMMAND:";
    for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
    cerr << endl;

    po::variables_map conf;
    InitCommandLine(argc, argv, &conf);
    WORD_DIM = conf["word_dim"].as<unsigned>();
    HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
    TAG_HIDDEN_DIM = conf["tag_hidden_dim"].as<unsigned>();

    unk_prob = conf["unk_prob"].as<float>();
    pdrop = conf["pdrop"].as<float>();

    DEBUG = conf.count("debug");

    assert(unk_prob >= 0.); assert(unk_prob <= 1.);
    assert(pdrop >= 0.); assert(pdrop <= 1.);

    vector<Instance> training,dev,test;
    string line;
    
    kUNK = wd.convert("*UNK*");
    //reading pretrained
    if(conf.count("pretrained")){
      cerr << "Loading from " << conf["pretrained"].as<string>() << " as pretrained embedding with" << WORD_DIM << " dimensions ... ";
      ifstream in(conf["pretrained"].as<string>().c_str());
      string word;
      unk_embedding.resize(WORD_DIM, 0);
      while(in>>word){
        vector<float> v(WORD_DIM);
        for(unsigned i = 0; i < WORD_DIM; i++) {in>>v[i]; unk_embedding[i] += v[i];}
        pretrained[wd.convert(word)] = v;
      }
      for(unsigned i = 0; i < WORD_DIM; i ++) unk_embedding[i] /= pretrained.size();
      cerr << pretrained.size() << " ok\n";
    }
    
    if(conf.count("lexicon")){
      cerr << "Loading from " << conf["lexicon"].as<string>() << " as lexion dictionary ...";
      ifstream in(conf["lexicon"].as<string>().c_str());
      string word;
      float v;
      while(in>>word){
    	in>>v;
	lex_dict[wd.convert(word)] = v;
      }
      cerr << lex_dict.size() << " ok\n";
    }

    //reading training data
    cerr << "Loading from " << conf["training_data"].as<string>() << "as training data : ";
    {
      ifstream in(conf["training_data"].as<string>().c_str());
      assert(in);
      while(getline(in, line)) {
        Instance instance;
        instance.load(line);
        training.push_back(instance);	
      }
      cerr<<training.size()<<"\n";
    }

    //couting
    set<unsigned> training_vocab;
    set<unsigned> singletons;
    {
      map<unsigned, unsigned> counts;
      for (auto& sent : training){
	const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<unsigned>& words = sent.words;
	for (unsigned i = 0; i < sent.size(); ++i){
	  if(pretrained.size() > 0){
	    if(pretrained.count(raws[i])) words[i] = raws[i];
	    else if(pretrained.count(lows[i])) words[i] = lows[i];
	  }
          training_vocab.insert(words[i]); counts[words[i]]++;
	}
      }
      for (auto wc : counts)
        if (wc.second == 1) singletons.insert(wc.first);
      
      cerr<<"the training word dict size is " << training_vocab.size()
	     << " where The singletons have " << singletons.size() << "\n";
    }

    //replace unk 
    {
      int unk = 0;
      int total = 0;
      for(auto& sent : training){
        for(auto& w : sent.words){
          if(singletons.count(w) && dynet::rand01() < unk_prob){
	  	w = kUNK;
		unk += 1;
 	  }
          total += 1;
        }
      }
      cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
    }

    //import lexicon score for each word
    {
      for(auto& sent : training){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
	vector<float>& lex_score = sent.lex_score;
	for(unsigned i = 0; i < sent.size(); ++i){
	  if(lex_dict.count(raws[i])) lex_score[i] = lex_dict[raws[i]];
	  else if(lex_dict.count(lows[i])) lex_score[i] = lex_dict[lows[i]];
	  else lex_score[i] = noscore;
        }
      }
    }

    //reading dev data 
    if(conf.count("dev_data")){
      cerr << "Loading from " << conf["dev_data"].as<string>() << "as dev data : ";
      ifstream in(conf["dev_data"].as<string>().c_str());
      string line;
      while(getline(in,line)){
        Instance inst;
        inst.load(line);
        dev.push_back(inst);
      }
      cerr<<dev.size()<<"\n";
    }
    //replace unk
    {
      int unk = 0;
      int total = 0;
      for(auto& sent : dev){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<unsigned>& words = sent.words;
	for(unsigned i = 0; i < sent.size(); i ++){
          if(pretrained.count(raws[i])) words[i] = raws[i];
	  else if(pretrained.count(lows[i])) words[i] = lows[i];
	  else if(training_vocab.count(raws[i])) words[i] = raws[i];
	  else{
	  	words[i] = kUNK;
		unk += 1;
	  }
          total += 1;
        }
      }
      cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
    }
 
    {
      for(auto& sent : dev){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<float>& lex_score = sent.lex_score;
        for(unsigned i = 0; i < sent.size(); ++i){
          if(lex_dict.count(raws[i])) lex_score[i] = lex_dict[raws[i]];
          else if(lex_dict.count(lows[i])) lex_score[i] = lex_dict[lows[i]];
          else lex_score[i] = noscore;
        }
      }
    }

    //reading test data
    if(conf.count("test_data")){
      cerr << "Loading from " << conf["test_data"].as<string>() << "as test data : ";
      ifstream in(conf["test_data"].as<string>().c_str());
      string line;
      while(getline(in,line)){
        Instance inst;
        inst.load(line);
        test.push_back(inst);
      }
      cerr<<test.size()<<"\n";
    }

    //replace unk
    {
      int unk = 0;
      int total = 0;
      for(auto& sent : test){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<unsigned>& words = sent.words;
        for(unsigned i = 0; i < sent.size(); i ++){
          if(pretrained.count(raws[i])) words[i] = raws[i];
          else if(pretrained.count(lows[i])) words[i] = lows[i];
          else if(training_vocab.count(raws[i])) words[i] = raws[i];
          else{
                words[i] = kUNK;
                unk += 1;
          }
          total += 1;
        }
      }
      cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
    }

    {
      for(auto&sent : test){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<float>& lex_score = sent.lex_score;
        for(unsigned i = 0; i < sent.size(); ++i){
          if(lex_dict.count(raws[i])) lex_score[i] = lex_dict[raws[i]];
          else if(lex_dict.count(lows[i])) lex_score[i] = lex_dict[lows[i]];
          else lex_score[i] = noscore;
        }
      }
    }

    VOCAB_SIZE = wd.size();

    ostringstream os;
    os << "lstmclassifier"
       << '_' << WORD_DIM
       << '_' << HIDDEN_DIM
       << '_' << LAYERS
       << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameter will be written to: " << fname << endl;
    float best = 0;

    Model model;
    Trainer* sgd = nullptr;
    unsigned method = conf["train_methods"].as<unsigned>();
    if(method == 0)
  	sgd = new SimpleSGDTrainer(&model,0.1, 0.1);
    else if(method == 1)
	sgd = new MomentumSGDTrainer(&model,0.01, 0.9, 0.1);
    else if(method == 2){
	sgd = new AdagradTrainer(&model);
	sgd->clipping_enabled = false;	
    }
    else if(method == 3){
	sgd = new AdamTrainer(&model);
  	sgd->clipping_enabled = false;
    } 
    LSTMClassifier lstmClassifier(model);

    if (conf.count("model")) {
    string fname = conf["model"].as<string>();
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
    }

if(DEBUG)	cerr<<"begin\n";
    unsigned report_every_i = conf["report_i"].as<unsigned>();
    unsigned dev_report_every_i = conf["dev_report_i"].as<unsigned>();
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int exceed_count = 0;
    unsigned count = 0;
    while(count < conf["count_limit"].as<unsigned>()) {
        Timer iteration("completed in");
        float loss = 0;
        unsigned ttags = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) {
                si = 0;
                if (first) {
                    first = false;
                }
                else {
                    sgd->update_epoch();
                    if (1) {
                        float acc = 0.f;
                        cerr << "\n***DEV [epoch=" << (lines / (float)training.size()) << "] ";
                        evaluate(dev, lstmClassifier, acc);
                        if (acc > best) {
                            best = acc;
                            cerr<< "Exceed" << " ";
                            float tacc = 0;
                            evaluate(test, lstmClassifier, tacc);
                            ofstream out(fname);
                            boost::archive::text_oarchive oa(out);
                            oa << model;
                            exceed_count ++;
                        }
			cerr<<"\n";
                    }
                }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                count++;
            }

            ComputationGraph cg;
            auto& sentx_y = training[order[si]];
	    float num_correct = 0;
            Expression nll= lstmClassifier.BuildGraph(sentx_y, cg, num_correct, true);
	    loss += as_scalar(cg.incremental_forward(nll));
            cg.backward(nll);
            sgd->update(1.0);
            ++si;
            ++lines;
            ++ttags;
        }
        sgd->status();
        cerr << " E = " << (loss / ttags) <<" "<<loss << "/"<<ttags<<" ";

        // show score on dev data?
        report++;
        continue;
        if ( report % dev_report_every_i == 1 ) {
            float acc = 0.f;
            cerr << "\n***DEV [epoch=" << (lines / (float)training.size()) << "] ";
            evaluate(dev, lstmClassifier, acc);
            if (acc > best) {
                best = acc;
                cerr<< "Exceed" << " ";
                float tacc = 0;
                evaluate(test, lstmClassifier, tacc);
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
                exceed_count++;
            }
	    cerr<<"\n";
        }
    }
    delete sgd;
}

