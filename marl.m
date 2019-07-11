agents = 5;
gamma = 0.5;
action_space = [1,2,3,4,5];
state_space = [1,2,3,4,5,6,7,8,9,10];
num_episodes = 100;
Q = zeros([length(state_space) agents length(action_space)]);
V = zeros([length(state_space) agents]);
reward = zeros(1,agents);
observed_reward = zeros(num_episodes,agents);
t=1;
fairness_value = zeros([num_episodes]);
state = randperm(length(state_space),1);
for i = 1:num_episodes
    action = policy(state,Q);
        for j = 1:agents
            if(action ~= j)
                reward(j) = 0;
            end
            if(action==j)
                reward(j) = state;
            end
        end
        state_new = randperm(length(state_space),1);
        observed_reward(t,:) = reward;
        V(state,:)=Q(state,action,:);
        at=alpha(t);
        for j =1:agents
            Q(state,action,j) = (1-at)*Q(state,action,j) + at*(reward(j) + gamma*V(state_new,j));
            long_run_avg_rwd = sum(observed_reward)/(t);
            fairness_value(t)=sum(log(1 + long_run_avg_rwd)/log(10));
        end
        %Q(state,action,:) = (1-at)*Q(state,action,:) + at*(reward + gamma*V(state_new,:))
        state = state_new;
        t=t+1 ;
        
    
end
index = [1:num_episodes];
plot(index,fairness_value);

function action = policy(state,Q)
q = Q(state,:,:);
q_new = squeeze(q);
q_log = log(1+q_new)/log(10);
q_final = sum(q_log,2);
[argvalue, argmax] = max(q_final);
action = argmax;
end

function at = alpha(t)
at = t^(2/3);
end
