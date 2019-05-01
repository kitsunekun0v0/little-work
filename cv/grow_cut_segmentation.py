def grow_cut(X, seeding):
    """
    X: input image
    seeding: seeding map, 1=foreground, 2=background
    """  
    # seeding points for grow cut
    # initialise the strength for each voxel
    dx,dy = X.shape
    label = np.zeros([dx,dy])
    strength = np.zeros([dx,dy])

    fg = np.where(seeding==1) # get seeding points location
    i = fg[0]
    j = fg[1]   
    label[i,j] = 1
    strength[i,j] = 1

    bg = np.where(seeding==2)
    i = bg[0]
    j = bg[1]
    label[i,j] = 3
    strength[i,j] = 1

    # start grow cut
    converged = False
    br  = np.where(seeding>0)
    while ~converged:
      # copy previous state
      label_next = label
      strength_next = strength
      converged = True

      for x in range(1:dx-1):
	for y in range(1:dy-1):
	    # neighbour label, strength and value. 3x3 matrics where [1,1] is the current
	    # cell. 
	    x = br[0][w]
	    y = br[1][w]

	    nl = label[x-1:x+2, y-1:y+2] #nl is the label of neighbour
	    ns = strength[x-1:x+2, y-1:y+2] #ns is strength of neighbour
	    nb = X[x-1:x+2, y-1:y+2] #nb is value of neightbour.

	    # neighbour try to attack current cell
	    g = 1-np.abs(nb-X[x,y])/np.max(np.abs(nb-X[x,y])) #monotonous decreasing function bound to [0,1]
	    af = g*ns #attack force of neighbours
	    af_max = np.amax(af) #the greatest attack force from neighbours

	    # if the strength of current cell is less than the greatest attack force,
	    # then replace the state of current cell with the state of highest attack force neighbour.
	    if af_max>strength[x,y]:
	        x1, y1 = np.unravel_index(af.argmax(), af.shape) #find win neighbour
		label_next[x,y] = nl[x1,y1]
		strength_next[x,y] = ns[x1,y1]
		converged = False #if update happened, not converged.

       label = label_next
       strength = strength_next
 
    return label
