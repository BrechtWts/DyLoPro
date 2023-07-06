""" General function called upon by every plotting function. """
def plt_period(x, y, axes, y_label, number=None, max_k=None, title=None, label=None,  location=None, color=None):
    """Generic plotting function. 4 options:

    #. Data needs to be plotted on a figure with 1 other plot, with each of 
       the 2 plots a different vertical axis (left and right). This data is 
       assigned to the left vertical axis. Needed arguments: the required 
       arguments, label = string, location = 'left', color = string.
    
    #. Data needs to be plotted on a figure with 1 other plot, with each of 
       the 2 plots a different vertical axis (left and right). This data is 
       assigned to the right vertical axis. Needed arguments: the required 
       arguments, label = string, location = 'right', color = string.
    
    #. Data needs to be plotted on a figure that will not contain any other 
       plots. Needed arguments: the required arguments, title = string.
    
    #. Data needs to be plotted on a figure that will contain several other 
       plots, but all of the same quantity / unit. Hence, only one vertical 
       axis (the left one) will be used by max_k plots. Needed arguments: 
       the required arguments, number = int, max_k = int, title = string, 
       label = string.

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    axes : _type_
        _description_
    y_label : _type_
        _description_
    number : _type_, optional
        _description_, by default None
    max_k : _type_, optional
        _description_, by default None
    title : _type_, optional
        _description_, by default None
    label : _type_, optional
        _description_, by default None
    location : _type_, optional
        _description_, by default None
    color : _type_, optional
        _description_, by default None
    """
    marker_list = ['o', 's', '+', 'x', 'D', 'P', '1', 'v', '*', '>', '<']
    num_markers = len(marker_list)
    
    if color:
        if location=='left': #1
            axes.plot(x, y, label=label, color = color, marker='o')
            axes.legend(loc='upper left')
            axes.grid(True, axis='x')
            axes.set_ylabel(y_label, color=color)
        elif location=='right': #2
            # axes.plot(x, y, label="DFR {}".format(number), color = color, marker='o')
            axes.plot(x, y, label=label, color = color, marker='s')
            axes.legend(loc='upper right')
            # axes.set_ylabel("Number of DFR {} per case".format(number), color=color)
            axes.set_ylabel(y_label, color=color)
        if title:
            axes.set_title(title)

    else:
        if label: #4
            # Marker_idx configured st in case of number > num_markers, 
            # we start again at the beginning of the marker list. 
            marker_idx = number - 1 - num_markers*((number-1)//num_markers)
            axes.plot(x, y, label=label, marker= marker_list[marker_idx])
            if number==max_k:
                if max_k > 2: 
                    axes.legend(loc="upper left", fontsize='x-small')
                else:
                    axes.legend()
                # axes.set_ylabel("# Directly-Follows Relations (DFR) per Case")
                axes.set_ylabel(y_label)
                axes.grid(True)
                # axes.set_title("{} evolution of #occurances/case for the {} most common Direclty-Follows Relationships (DFRs)".format(frequency, max_k))
                axes.set_title(title)
        else: #3
            axes.plot(x, y, marker='o')
            # axes.set_ylabel("Number of DFR {} per case".format(number))
            axes.set_ylabel(y_label)
            axes.grid(True)
            # axes.set_title("DFR {}: {} evolution of #occurances/case".format(number, frequency))
            axes.set_title(title)