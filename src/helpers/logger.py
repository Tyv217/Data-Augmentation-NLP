class Logger():
    def log_batch(self, epoch_number, index, total_batches, accuracy, count):
        print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f} |'.format(epoch_number, index, total_batches, accuracy / count))        

    def log_epoch(self, epoch_number, elapsed_time, accuracy):
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch_number, elapsed_time, accuracy))
        print('-' * 59)
    
    def log_test_result(self, test_accuracy):
        print('The results of test dataset:')
        print('test accuracy {:8.3f}'.format(test_accuracy))