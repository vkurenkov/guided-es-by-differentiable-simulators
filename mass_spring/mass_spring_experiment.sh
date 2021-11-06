for i in {0..5}; do
    # Run brute-froce gd on simulator with time-of-impact fix
    python mass_spring.py --alg="taichi" --seed=$i --iterations=2000 --use-toi

    # Run brute-froce gd on simulator without time-of-impact fix
    python mass_spring.py --alg="taichi" --seed=$i --iterations=2000
    
    # Run vanilla es on simulator without time-of-impact fix
    python mass_spring.py --alg="guided-es" --seed=$i --iterations=5000

    # Run guided es on simulator without time-of-impact fix
    python mass_spring.py --alg="vanilla-es" --seed=$i --iterations=10000
done