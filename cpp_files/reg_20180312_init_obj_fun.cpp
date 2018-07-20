// 
// dyn.unload('cpp_files/reg_20180312_init_obj_fun.dll')
// compile('cpp_files/reg_20180312_init_obj_fun.cpp')
// dyn.load('cpp_files/reg_20180312_init_obj_fun.dll')
#include <TMB.hpp>


template<class Type>
Type get_prob_of_scoreline(Type me, Type you, Type prob_win_point){
    
    Type out_prob;
    
    vector<Type> tmp_vec(2); // it's annoying, but we need to do this to get pairwise minimum
    tmp_vec[0] = me;
    tmp_vec[1] = you;
    Type min_me_you = min(tmp_vec);

    
    if (you >= 20) {
        out_prob = nchoosek(Type(40), Type(20)) * pow(2, min_me_you - 20) * pow(prob_win_point, me) * pow(1 - prob_win_point, you);
        
    } else {
        out_prob = nchoosek(Type(20) + min_me_you, min_me_you) * pow(prob_win_point, me) * pow(1 - prob_win_point, you);
        
    }
    
    return(out_prob);
}



template<class Type>
Type objective_function<Type>::operator() () {
    
    
    
    DATA_VECTOR(you);
    DATA_VECTOR(me);
    DATA_VECTOR(score_diff);
    
    
    // parameters:
    PARAMETER(p_0);
    PARAMETER(alpha_0);
    PARAMETER(alpha_1);
    PARAMETER(beta);
    
    int n_obs = me.size();
    
    Type p = p_0; // initialise initial model fitted probability
    Type nll = 0; // initialise model negloglik
    vector<Type> p_vec_out(n_obs);
    
    for (int i = 0; i < n_obs; i++) {
        
        if (i > 0) {
            // mixing coefficient
            Type alpha = invlogit(alpha_0 + alpha_1 * score_diff[i - 1]);
            
            // update p with model
            p =  alpha * p_0 + (1 - alpha) *  invlogit(logit(p) + beta);
            
            
        }
        nll += -log(get_prob_of_scoreline(me[i], you[i], p));
        // store into vector to be returned to R
        p_vec_out[i] = p;
    }
    
    
    REPORT(p_vec_out);
    return nll;
    
}