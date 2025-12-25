# TODO

## Data Handling & Row Dropping

### Locations where rows are dropped:

1. **bid_optimization.py - create_feature_matrix()**: Line ~207
   - Drops keyword-match type-region combinations not found in training data (after merge_asof)
   - Uses `merge_asof` to find nearest date match for exact (Keyword, Match type, Region) combinations
   - **TODO**: Monitor how many combinations are dropped. If too many, consider:
     - Using broader regions or match type categories as fallback
     - Interpolating missing values from nearby dates
     - Using default/zero values for sparse combinations

2. **tidy_get_data.py - main()**: Line ~63
   - Removes rows with ANY NaN values across all columns
   - **TODO**: Review if this is too aggressive
   - **TODO**: Consider which columns actually need non-null values for model training
   - **TODO**: Check if this impacts data quality and quantity for model training

### Approximate Matching Strategy

Currently using `merge_asof` for nearest-date matching:
- Exact match on: Keyword, Match type, Region
- Nearest date match direction: 'nearest'
- **TODO**: Consider alternative strategies:
  - Use `direction='backward'` to only use historical data
  - Use `tolerance` parameter to limit how far back we go
  - Add fallback logic for combinations with no nearby matches
  - Log statistics on which combinations matched which dates

## Model & Optimization

- Verify that matched features from different dates still make sense for predictions
- Consider impact of date mismatch on model accuracy
- **TODO**: Add feature indicating how far the matched date is from target date

## Testing

- **TODO**: Add unit tests for merge_asof matching logic
- **TODO**: Validate that data quality isn't degraded by row dropping
- **TODO**: Monitor distribution of dropped vs. kept rows
