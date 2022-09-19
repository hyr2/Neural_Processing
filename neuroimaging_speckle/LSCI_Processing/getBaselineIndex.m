function idx = getBaselineIndex(t)
%getBaselineIndex Get baseline indicies based on user-defined timespan
% getBaselineIndex(t) Prompts the user to define the baseline timespan and 
% returns the corresponding index range according to the data timing.
%

  t1 = input('Enter baseline start time (s): ');
  idx_1 = find(t > t1, 1);
  t2 = input('Enter baseline duration (s): ');
  idx_2 = find(t > t1 + t2, 1) - 1;
  idx = idx_1:idx_2;
  fprintf('Defining baseline from %d frames\n', length(idx));

end
