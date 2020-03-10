call plug#begin('D:/softwares/Neovim/plugged')

Plug 'scrooloose/nerdtree', { 'on': 'NERDTreeToggle' }
Plug 'tpope/vim-commentary'
Plug 'jiangmiao/auto-pairs'
Plug 'Yggdroot/LeaderF', { 'do': '.\install.bat'  }
Plug 'easymotion/vim-easymotion'
Plug 'mhinz/vim-startify'
Plug 'tomlion/vim-solidity'

Plug 'ncm2/ncm2'
Plug 'roxma/nvim-yarp'
Plug 'ncm2/ncm2-bufword'
Plug 'ncm2/ncm2-path'
Plug 'ncm2/ncm2-jedi', {'for': 'Python'}
Plug 'ncm2/ncm2-tern',  {'do': 'npm install', 'for': 'Javascript'}
Plug 'mhartington/nvim-typescript', {'for': 'Typescript', 'do': 'npm i -g typescript'}
Plug 'ncm2/ncm2-go', {'for': 'Go'}
Plug 'ncm2/ncm2-html-subscope'
Plug 'ncm2/ncm2-markdown-subscope'
call plug#end()

autocmd BufEnter * call ncm2#enable_for_buffer()
set completeopt=noinsert,menuone,noselect
inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"

map <Leader>l :Commentary<CR>
map <C-_> :Commentary<CR>


map <Leader>r :call CompileRun()<CR>
func! CompileRun()
    exec "w" 
    if &filetype == 'c' 
        exec '!g++ % -o %<'
        exec '!time ./%<'
    elseif &filetype == 'cpp'
        exec '!g++ % -o %<'
        exec '!time ./%<'
    elseif &filetype == 'python'
        exec '!python %'
    elseif &filetype == 'sh'
        :!bash %
	elseif &filetype == 'go'
        exec '!go run %'
    endif                                                                              
endfunc


set nu
set tabstop=4
set softtabstop=4
set shiftwidth=4
set expandtab
set smartindent

autocmd FileType javascript set tabstop=2 softtabstop=2 shiftwidth=2
autocmd FileType html set tabstop=2 softtabstop=2 shiftwidth=2
autocmd FileType css set tabstop=2 softtabstop=2 shiftwidth=2

" jump to the last position when re-opening a file
if has("autocmd")
  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif

