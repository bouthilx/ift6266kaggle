def experiment(state,channel):

    (best_validation_loss, test_score, minutes_trained, iter) = \
        sgd_optimization_mnist(state.learning_rate, state.n_hidden, state.L1_reg,
            state.L2_reg, state.batch_size, state.n_iter)

    state.best_validation_loss = best_validation_loss
    state.test_score = test_score
    state.minutes_trained = minutes_trained
    state.iter = iter
    return channel.COMPLETE
