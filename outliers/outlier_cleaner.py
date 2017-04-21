#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    
    cleaned_data = [(age, worth, (worth-pred)**2.0) for (age,worth,pred) in zip(ages, net_worths, predictions)]
    cleaned_data = sorted(cleaned_data, key=lambda point: point[2])

    ### your code goes here
    
    return cleaned_data[:9*len(cleaned_data)//10]

