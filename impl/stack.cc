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
unsigned DOMAIN_N = 4;
unsigned ATTENTION_HIDDEN_DIM = 100;
float noscore = 10000;
dynet::Dict wd;

int kUNK; //tzy
unordered_map<unsigned, vector<float> > pretrained;
vector<float> unk_embedding;

unordered_map<unsigned, float> lex_dict;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
	("training_data,T", po::value<string>(), "target domain training data")
	("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data", po::value<string>(), "Test corpus")
	("domain0",po::value<string>(), "domain 0")
	("domain1",po::value<string>(), "domain 1")
	("domain2",po::value<string>(), "domain 2")
	("domain3",po::value<string>(), "domain 3")
        ("indomain_i",po::value<unsigned>()->default_value(2), "stack training rounds")
	("indomain_round",po::value<unsigned>()->default_value(5), "one domain in stack training rounds")
	("pdrop", po::value<float>()->default_value(0.5), "dropout probabilty")
	("unk_prob,u", po::value<float>()->default_value(0.1), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("word_dim", po::value<unsigned>()->default_value(200), "word embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(150), "hidden dimension")
        ("tag_hidden_dim", po::value<unsigned>()->default_value(64), "tag hidden dimension")
	("layers", po::value<unsigned>()->default_value(1), "layers")
	("train,t", "Should training be run?")
        ("pretrained,w", po::value<string>(), "Pretrained word embeddings")
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
  if ( conf->count("dev_data") == 0 || conf->count("test_data") == 0) {
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
	unsigned label;
	
	Instance(){clear();};
        ~Instance(){};
	void clear(){
		raws.clear();
		lows.clear();
		words.clear();
	}	
	friend ostream& operator << (ostream& out, Instance& instance){
		for(unsigned i = 0; i < instance.raws.size(); i ++){
			out << wd.convert(instance.raws[i]) << "/"
			    << wd.convert(instance.lows[i]) << "/";
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
        }
	unsigned size(){assert(raws.size() == lows.size()); return raws.size();}
};

struct LSTMClassifier {
    LookupParameter p_word;

    LookupParameter p_l2rR;
    LookupParameter p_r2lR;
    LookupParameter p_bias;

    LookupParameter p_lbias;
    LookupParameter p_tag2label;

    //Parameter p_attbias;
    //Parameter p_input2att;
    //Parameter p_att2attexp;

    //Parameter p_R;
    //Parameter p_bias_t;
    Parameter p_feat2label;
    Parameter p_lbias_t;

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

        p_l2rR = model.add_lookup_parameters(DOMAIN_N,{TAG_HIDDEN_DIM, HIDDEN_DIM});
        p_r2lR = model.add_lookup_parameters(DOMAIN_N,{TAG_HIDDEN_DIM, HIDDEN_DIM});
	p_bias = model.add_lookup_parameters(DOMAIN_N,{TAG_HIDDEN_DIM});

	p_tag2label = model.add_lookup_parameters(DOMAIN_N,{2, TAG_HIDDEN_DIM});
	p_lbias = model.add_lookup_parameters(DOMAIN_N,{2});

	//p_attbias = model.add_parameters({ATTENTION_HIDDEN_DIM});
        //p_input2att = model.add_parameters({ATTENTION_HIDDEN_DIM, TAG_HIDDEN_DIM + 2});
        //p_att2attexp = model.add_parameters({ATTENTION_HIDDEN_DIM});
	
	//p_R = model.add_parameters({TAG_HIDDEN_DIM, TAG_HIDDEN_DIM+2});
        //p_bias_t = model.add_parameters({TAG_HIDDEN_DIM});

        p_feat2label = model.add_parameters({2, DOMAIN_N * 2});
        p_lbias_t = model.add_parameters({2});

        p_start = model.add_parameters({WORD_DIM});
	p_end = model.add_parameters({WORD_DIM});

        for(auto& it : pretrained){
	    p_word.initialize(it.first, it.second);
        }
    }

    // return Expression of total loss
    Expression BuildGraph_domain(Instance& inst, ComputationGraph& cg, float& num_correct, unsigned index) {
        const vector<unsigned>& sent = inst.words;
	unsigned label = inst.label;
        const unsigned slen = sent.size() ;

        l2rbuilder.new_graph(cg);  // reset builder for new graph
        l2rbuilder.start_new_sequence();

        r2lbuilder.new_graph(cg);  // reset builder for new graph
        r2lbuilder.start_new_sequence();

	Expression i_l2rR = lookup(cg, p_l2rR, index);
	Expression i_r2lR = lookup(cg, p_r2lR, index);
	Expression i_bias = lookup(cg, p_bias, index);
        
	Expression i_tag2label = lookup(cg, p_tag2label, index);
	Expression i_lbias = lookup(cg, p_lbias, index);

	Expression word_start = parameter(cg, p_start);
        Expression word_end = parameter(cg, p_end);

        vector<Expression> i_words(slen);
        for (unsigned t = 0; t < slen; ++t) {
            i_words[t] = lookup(cg, p_word, sent[t]);
            i_words[t] = dropout(i_words[t], pdrop);
        }

        l2rbuilder.add_input(word_start);
        r2lbuilder.add_input(word_end);
        for (unsigned t = 0; t < slen; ++t) {
	    l2rbuilder.add_input(i_words[t]);
            r2lbuilder.add_input(i_words[slen - 1 - t]);
        }
	l2rbuilder.add_input(word_end);
        r2lbuilder.add_input(word_start);

        Expression i_r =  tanh(i_bias + i_l2rR * l2rbuilder.back() + i_r2lR * r2lbuilder.back());
	Expression i_r_t = i_lbias + i_tag2label * i_r;
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
    Expression BuildGraph(Instance& inst, ComputationGraph& cg, float& num_correct, bool train) {
        const vector<unsigned>& sent = inst.words;
	unsigned label = inst.label;
        const unsigned slen = sent.size() ;

        l2rbuilder.new_graph(cg);  // reset builder for new graph
        l2rbuilder.start_new_sequence();

        r2lbuilder.new_graph(cg);  // reset builder for new graph
        r2lbuilder.start_new_sequence();

	Expression word_start = parameter(cg, p_start);
        Expression word_end = parameter(cg, p_end);

if(DEBUG)	cerr<<"sent size " << slen<<"\n";
        vector<Expression> i_words(slen);
        for (unsigned t = 0; t < slen; ++t) {
            i_words[t] = lookup(cg, p_word, sent[t]);
            if(train) i_words[t] = dropout(i_words[t], pdrop);
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

	vector<Expression> inputs;
	for(unsigned index = 0; index < DOMAIN_N; ++index){

	Expression i_l2rR = lookup(cg, p_l2rR, index);
	Expression i_r2lR = lookup(cg, p_r2lR, index);
	Expression i_bias = lookup(cg, p_bias, index);
        
	Expression i_tag2label = lookup(cg, p_tag2label, index);
	Expression i_lbias = lookup(cg, p_lbias, index);

        Expression i_r =  tanh(i_bias + i_l2rR * l2rbuilder.back() + i_r2lR * r2lbuilder.back());
	Expression i_r_t = softmax(i_lbias + i_tag2label * i_r);
	
	//inputs.push_back(concatenate({i_r,i_r_t}));
	inputs.push_back(i_r_t);
	}
if(DEBUG)	cerr<<"domain ok\n";
	Expression cross_feature = concatenate(inputs);
	//Expression i_bias_t = parameter(cg, p_bias_t);
	//Expression i_R = parameter(cg, p_R);
	Expression i_lbias_t = parameter(cg, p_lbias_t);
	Expression i_feat2label = parameter(cg, p_feat2label);

	/*Expression attbias = parameter(cg, p_attbias);
	Expression input2att = parameter(cg, p_input2att);
	Expression att2attexp = parameter(cg, p_att2attexp);	
	//attention	
	vector<Expression> att(inputs.size());
        for(unsigned t = 0; t < inputs.size(); t ++){
                att[t] = tanh(affine_transform({attbias, input2att, inputs[t]}));
        }
        Expression att_col = transpose(concatenate_cols(att));
        Expression attexp = softmax(att_col * att2attexp);

        Expression input_col = concatenate_cols(inputs);
        Expression att_pool = input_col * attexp;
	*/
	//Expression cross_i_r = tanh(i_bias_t + i_R * att_pool);
	//Expression cross_i_r_t = i_lbias_t + i_feat2label * cross_i_r;
	Expression cross_i_r_t = i_lbias_t + i_feat2label * cross_feature;
	Expression output_loss = pickneglogsoftmax(cross_i_r_t, label);
if(DEBUG)	cerr<<"target done\n";
        auto prob_value = as_vector(cg.incremental_forward(cross_i_r_t));
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

    vector<vector<Instance> > training_domain(DOMAIN_N);
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
    
    //reading training data
    {
	{
	cerr << "Loading from " << conf["training_data"].as<string>() << "as target domain training data : ";
        ifstream in(conf["training_data"].as<string>().c_str());
        assert(in);
        while(getline(in, line)) {
                Instance instance;
                instance.load(line);
                training.push_back(instance);
        }
        cerr<<training.size()<<"\n";
	}

	{
      	cerr << "Loading from " << conf["domain0"].as<string>() << "as domain0 data : ";
       	ifstream in(conf["domain0"].as<string>().c_str());
      	assert(in);
      	while(getline(in, line)) {
        	Instance instance;
        	instance.load(line);
        	training_domain[0].push_back(instance);	
      	}
      	cerr<<training_domain[0].size()<<"\n";
        }

	{
	cerr << "Loading from " << conf["domain1"].as<string>() << "as domain1 data : ";
        ifstream in(conf["domain1"].as<string>().c_str());
        assert(in);
        while(getline(in, line)) {
                Instance instance;
                instance.load(line);
                training_domain[1].push_back(instance);
        }
        cerr<<training_domain[1].size()<<"\n";
	}

	{
	cerr << "Loading from " << conf["domain2"].as<string>() << "as domain2 data : ";
        ifstream in(conf["domain2"].as<string>().c_str());
        assert(in);
        while(getline(in, line)) {
                Instance instance;
                instance.load(line);
                training_domain[2].push_back(instance);
        }
        cerr<<training_domain[2].size()<<"\n";
	}

	{
	cerr << "Loading from " << conf["domain3"].as<string>() << "as domain3 data : ";
        ifstream in(conf["domain3"].as<string>().c_str());
        assert(in);
        while(getline(in, line)) {
                Instance instance;
                instance.load(line);
                training_domain[3].push_back(instance);
        }
        cerr<<training_domain[3].size()<<"\n";
	}
    }
    
    //couting
    set<unsigned> training_vocab;
    set<unsigned> singletons;
    {
      map<unsigned, unsigned> counts;
      //target domain
      /*for (auto& sent : training){
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
      }*/
      //cross domain
      for (unsigned t = 0; t < DOMAIN_N; t ++){
      for (auto& sent : training_domain[t]){
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
      }
      for (auto wc : counts)
        if (wc.second == 1) singletons.insert(wc.first);
      
      cerr<<"the training word dict size is " << training_vocab.size()
	     << " where The singletons have " << singletons.size() << "\n";
    }

    //replace unk 
    {
      //target domain
      int unk = 0;
      int total = 0;
      for(auto& sent : training){
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
      //cross domain
      for(unsigned t = 0; t < DOMAIN_N; t ++){
      int unk = 0;
      int total = 0;
      for(auto& sent : training_domain[t]){
        for(auto& w : sent.words){
          if(singletons.count(w) && dynet::rand01() < unk_prob){
	  	w = kUNK;
		unk += 1;
 	  }
          total += 1;
        }
      }
      cerr << "domain "<< t <<" the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
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

    VOCAB_SIZE = wd.size();

    ostringstream os;
    os << "lstmclassifier"
       << '_' << WORD_DIM
       << '_' << HIDDEN_DIM
       << '_' << LAYERS
       << "_i" << conf["indomain_i"].as<unsigned>()
       << "_r" << conf["indomain_round"].as<unsigned>()
       << "-pid" << getpid() << ".pretrained.params";
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

    unsigned report_every_i = conf["report_i"].as<unsigned>();
    unsigned dev_report_every_i = conf["dev_report_i"].as<unsigned>();

    unsigned indomain_i = conf["indomain_i"].as<unsigned>();
    unsigned indomain_round = conf["indomain_round"].as<unsigned>();
//training =================================================
//
//
    //individual domain training
/*    {
	for(unsigned i = 0; i < indomain_round; i++){
	cerr<< "round " << i << "\n";
	for(unsigned index = 0; index < DOMAIN_N; index++){
		vector<unsigned> order;
		for(unsigned t = 0; t < training_domain[index].size(); t ++) order.push_back(t);
		for(unsigned j = 0; j < indomain_i; j ++){
		float total_loss = 0.0;
		float loss = 0.0;
		float num_correct = 0;
		shuffle(order.begin(), order.end(), *rndeng);	
		cerr<<" Domain " << index << "\n"; 
		for(unsigned t = 0; t < training_domain[index].size(); t ++){
    			ComputationGraph cg;
        		Expression nll= lstmClassifier.BuildGraph_domain(training_domain[index][order[t]],
									cg,
									num_correct,
									index);
        		float tmp = as_scalar(cg.incremental_forward(nll));
        		total_loss += tmp;
			loss += tmp;
			if(t != 0 && t % 500 == 0){
				sgd->status();
				cerr << "E = " << loss / 500 <<"\n";
				loss = 0;
			}
			cg.backward(nll);
        		sgd->update(1.0);
		}
		sgd->update_epoch();
		cerr << "Domain "<<index <<
			" E = " << total_loss / training_domain[index].size() <<
			" acc " << num_correct/training_domain[index].size() << "\n";
		}
	}
	}
    } 

    ofstream out(fname);
    boost::archive::text_oarchive oa(out);
    oa << model; 

    delete sgd;
    return 0;
*/
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int exceed_count = 0;
    unsigned count = 0;
    unsigned si = training.size();
    vector<unsigned> order;
    for(unsigned t = 0; t < training.size(); t ++) order.push_back(t);

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

