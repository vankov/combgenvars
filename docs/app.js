(function(angular) {
	'use strict';

	angular.module('varsApp', ['ui.bootstrap', 'googlechart'])

	.controller('varsController', function($timeout, $rootScope, $http, $scope, $filter, $q) {
		$scope.vars_config = {
			sem_dim: 20,
			n_tokens: 7,
			max_arity: 2
		};

		$scope.propositions = [];
		$scope.symbols = [];		
		$scope.vars = {};

		$scope.selectedProposition = null;

		this.$onInit = function() {
			$scope.createPropositions();
		}

		$scope.resetVars = function(vars){
			vars.semnatics = {};
			vars.structure = {};

			var sem = {};
			var struct = {};

			for(var i = 0; i < $scope.vars_config.n_tokens; i++) {

				sem[i] = [];
				struct[i] = [];

				for (var j = 0; j < $scope.vars_config.sem_dim; j++) {
					sem[i].push(0);
				}

				for (var a = 0; a < $scope.vars_config.max_arity; a++){
					struct[i][a] = [];
					for(var k = 0; k < $scope.vars_config.n_tokens; k++) {
						struct[i][a].push(0);
					}
				}
			}

			vars.semantics = sem;
			vars.structure = struct;		

		}

		$scope.resetVars($scope.vars);

		$scope.stm = {

		}

		$scope.getToken = function () {
			var token = -1;
			if (Object.keys($scope.stm).length >= $scope.vars_config.n_tokens) {
				throw("STM Full !");
				return -1;
			}
			while(token == -1) 
			{
				token = Math.floor(Math.random() * $scope.vars_config.n_tokens);
				if ($scope.stm[token]) {
					token = -1;
				}
			}

			return token;
		}

		$scope.addToStm = function (symbol) {
			var token = $scope.getToken();
			$scope.stm[token] = symbol;
		}

		$scope.getAssignedToken = function(symbol) {
			var symbol_token = -1;

			angular.forEach($scope.stm, function(symbol2, token){
				if (symbol2.id == symbol.id) {
					symbol_token = token;
				}
			});

			return symbol_token;
		}

		$scope.$watchCollection('vars_config', function(sem_dim) {
			$scope.resetVars($scope.vars);
			$scope.stm = {};
		});

		$scope.$watchCollection('stm', function(sem_dim) {
			angular.forEach($scope.stm, function(item, token){
				$scope.encoder.populateEmbedding(item.concept, token);
				angular.forEach(item.args, function(arg, i){
					var args = [];
					if (arg.length > 0) {
						angular.forEach(arg, function(a){
							args.push(a);
						});
					} else {
						args.push(arg);
					}
					angular.forEach(args, function(a){
						var assigned_token = $scope.getAssignedToken(a);
						if (assigned_token >= 0){
							$scope.vars.structure[token][i][assigned_token] = 1;
						};						
					})
				});				
			});
		});

		$scope.range = function(min, max, step) {
			step = step || 1;
			var input = [];
			for (var i = min; i < max; i += step) {
				input.push(i);
			}
			return input;
		};		

		$scope.getUnitStyle = function(act, useColor, isSemantic) {
						
			var color = $scope.encoder.getColor(act);

			var style = {
				'background-color': 'rgb(' + color + ',' + color + ',' + color + ')',
				'width': '15px',
				'height': '15px',
				'float': 'left',
				'margin-right': '2px',
				'border': '1px solid black',
			};

			if (useColor == false) {
				style['background-color'] = '';
				
			}

			if (isSemantic == false) {
				if (useColor) {
					style['background-color'] = act > 0 ? 'black' : '';
				} else {
					style['border'] = '';
				}
				style['width'] = style['height'];
				//style['magin'] = '';
				style['margin-left'] = '5px';
			}
			return style;
		}

		$scope.highlightedToken = -1;

		$scope.highlightToken = function($event, argNo, token, unitAct) {

			$event.stopPropagation();
			if (($scope.highlightedToken != token) && (unitAct > 0)) {
				$scope.highlightedToken = token;
			} else {
				$scope.highlightedToken = -1;
			}
			
		}

		$scope.getTokenStyle = function(token) {
			var style = {
			}

			if ($scope.highlightedToken == token) {
				style['background-color'] = 'red';
			}

			return style;
		}

		$scope.makeSymbol = function(concept, args) {
			var symnbol_no = 0;

			angular.forEach($scope.symbols, function(value, key) {
				if (value.concept == concept) {
					symnbol_no++;
				}
			  });

			var new_symbol = {
					"id": concept + "-" + symnbol_no,
					"concept": concept,
					"args": args
				}
			
			$scope.symbols.push(new_symbol);

			return new_symbol;
		}

		$scope.populateSTM = function() {
			if ($scope.selectedProposition) {
				if ($scope.selectedProposition.min_arity > $scope.vars_config.max_arity) {
					$scope.vars_config.max_arity = $scope.selectedProposition.min_arity;
					$timeout(function() {$scope.populateSTM()}, 500);
					return;
				}

				if ($scope.selectedProposition.min_n_tokens > $scope.vars_config.n_tokens) {
					$scope.vars_config.n_tokens = $scope.selectedProposition.min_n_tokens;
					$timeout(function() {$scope.populateSTM()}, 500);
					return;
				}				

				$scope.stm = {};
				$scope.resetVars($scope.vars);

				angular.forEach($scope.selectedProposition.symbols, function(symbol){
					$scope.addToStm(symbol);
				})

			}

		}

		$scope.createPropositions = function() {

			//cat and dog
			var cat = $scope.makeSymbol("cat", []);			
			var dog = $scope.makeSymbol("dog", []);
			var chases = $scope.makeSymbol("chases", [dog, cat]);
			var hates = $scope.makeSymbol("hates", [dog, cat]);			
			var because = $scope.makeSymbol("because", [hates, chases]);						

			$scope.propositions.push({
					symbols: [
						cat, dog, chases, hates, because
					],
					description: "The dog chases the cat because it hates it",
					min_arity: 2,
					min_n_tokens: 5
				});

			//John and Mary
			var john = $scope.makeSymbol("John", []);			
			var mary = $scope.makeSymbol("Mary", []);
			var loves = $scope.makeSymbol("loves", [john, mary]);

			$scope.propositions.push({
				symbols: [
					john, mary, loves
				],
				description: "John loves Mary",
				min_arity: 2,
				min_n_tokens: 3
			});

			var loves = $scope.makeSymbol("loves", [mary, john]);
			$scope.propositions.push({
				symbols: [
					john, mary, loves
				],
				description: "Mary loves John",
				min_arity: 2,
				min_n_tokens: 3
			});

			//Objects with color and shape
			var object1 = $scope.makeSymbol("object", []);			
			var object2 = $scope.makeSymbol("object", []);
			var blue = $scope.makeSymbol("blue", [object1]);
			var green = $scope.makeSymbol("green", [object2]);			
			var shape1 = $scope.makeSymbol("shape", [object1]);
			var shape2 = $scope.makeSymbol("shape", [object2]);
			var same = $scope.makeSymbol("same", [[shape1, shape2]]);

			$scope.propositions.push({
				symbols: [
					object1, object2, blue, green, shape1, shape2, same
				],
				description: "The blue object has the same as the green one",
				min_arity: 1,
				min_n_tokens: 7
			});

			//Rutherford analogy
			var electrons = $scope.makeSymbol("electrons", []);			
			var nucleus = $scope.makeSymbol("nucleus", []);
			var planets = $scope.makeSymbol("planets", []);
			var sun = $scope.makeSymbol("sun", []);
			var rotate1 = $scope.makeSymbol("rotate", [electrons, nucleus]);
			var rotate2 = $scope.makeSymbol("rotate", [planets, sun]);
			var same = $scope.makeSymbol("same", [[rotate1, rotate2]]);

			$scope.propositions.push({
				symbols: [
					electrons, nucleus, planets, sun, rotate1, rotate2, same
				],
				description: "Rutherford analogy, electrons:nucleus::planets:sun",
				min_arity: 2,
				min_n_tokens: 7
			});

		}

		$scope.vocabulary = {};

		$scope.random_distributed = {
			n_units: 20,
			reset: function() {

			},			
			getColor: function(act) {
				return Math.floor((1 - act) * 255);
			},
			populateEmbedding: function(word, token) {

				if ($scope.vocab[word]){
					$scope.vars.semantics[token] = $scope.vocab[word];
					return;
				}

				var v = [];
				for (var j = 0; j < $scope.vars_config.sem_dim; j++) {
					v.push(Math.random());
				}
				$scope.vocab[word] = v;

				$scope.vars.semantics[token] = v;		
			}
		}

		$scope.random_localist = {
			used: [],
			n_units: 30,
			reset: function() {
				$scope.random_localist.used = [];
			},		
			getColor: function(act) {
				return act > 0 ? 0: 255;
			},
			populateEmbedding: function(word, token) {

				if ($scope.vocab[word]){
					$scope.vars.semantics[token] = $scope.vocab[word];
					return;
				}

				var v = [];
				for (var j = 0; j < $scope.vars_config.sem_dim; j++) {
					v.push(0);
				}
				var el = -1;
				do
				{
					el = Math.floor(Math.random() * $scope.vars_config.sem_dim);
				}
				while($scope.random_localist.used.includes(el));

				v[el] = 1;
				$scope.random_localist.used.push(el);
				$scope.vocab[word] = v;

				$scope.vars.semantics[token] = v;
			}
		}

		$scope.glove = {
			n_units: 50,
			getColor: function(act) {
				if (act == 0) {
					return 255;
				} else {
					return Math.floor((1 - (act + 5.4593) / (5.4593 + 5.3101)) * 255);
				}
				
			},
			populateEmbedding: function(word, token) {

				if ($scope.vocab[word]){
					$scope.vars.semantics[token] = $scope.vocab[word];
					return;
				}

				$http.get("glove.php?word=" + word)
					.then(function(response) {
						var v = 						
							response.data.map(Number);
						$scope.vocab[word] = v;
						$scope.vars.semantics[token] = v;						
					}, function(error) {
						throw("Can't get Glove embedding for " + word);
					});

			}
		}

		$scope.$watch("new_encoder", function(new_encoder){
			$scope.vocab = {};
			switch(new_encoder) {
				case "localist":
					$scope.random_localist.reset();
					$scope.encoder = $scope.random_localist;
				break;
				case "distributed":
					$scope.encoder = $scope.random_distributed;
				break;				
				case "glove":
					$scope.encoder = $scope.glove;
				break;				

			}
			$scope.vars_config.sem_dim = $scope.encoder.n_units;
		});

		$scope.new_encoder = "distributed";

	});
}(window.angular));
