def printData(data):
    # epoch = data[0]
    # rmse_loss = data[1]
    
    with open('epochs.txt', 'a') as efile:
        with open('rmse_loss.txt', 'a') as rmsefile:
                efile.write(str(data[0]) + "\n")
                rmsefile.write(str(data[1]) + "\n")
    